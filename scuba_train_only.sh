#!/bin/bash
# ============================================================
#  SOVEREIGN SCUBA — TRAINING ONLY — CLEAN FINAL VERSION
#  torch==2.11, transformers==5.5.4, trl==1.2.0
#  peft==0.19.1, accelerate==1.13.0, bitsandbytes==0.49.2
# ============================================================

set -eo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKSPACE="/home/sesterce/sovereign_scuba_workspace"
STUDENT_PATH="$WORKSPACE/student"
TEACHER_PATH="$WORKSPACE/teacher"
DATASET_PATH="$WORKSPACE/scuba_bible.jsonl"
SFT_OUT="$WORKSPACE/sft_out"
DPO_OUT="$WORKSPACE/dpo_out"
MERGED_OUT="$WORKSPACE/merged"

mkdir -p "$SFT_OUT" "$DPO_OUT" "$MERGED_OUT"

BIBLE_COUNT=$(wc -l < "$DATASET_PATH" 2>/dev/null || echo 0)
echo "[OK] Bible: $BIBLE_COUNT samples found."
echo "[OK] Starting training pipeline..."

# ── PHASE 1: DISTILLATION ─────────────────────────────────────
echo "============================================================"
echo " [1/4] DISTILLATION"
echo "============================================================"
python3 << PYEOF
import os, torch, json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F

TEACHER_PATH = "/home/sesterce/sovereign_scuba_workspace/teacher"
STUDENT_PATH = "/home/sesterce/sovereign_scuba_workspace/student"
DATASET_PATH = "/home/sesterce/sovereign_scuba_workspace/scuba_bible.jsonl"
SFT_OUT      = "/home/sesterce/sovereign_scuba_workspace/sft_out"
ALPHA_KL=0.7; ALPHA_CE=0.3; TEMPERATURE=2.0

tok = AutoTokenizer.from_pretrained(STUDENT_PATH, use_fast=True)
tok.pad_token = tok.eos_token
tok.padding_side = "right"

print("[DISTILL] Loading teacher NF4...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
teacher = AutoModelForCausalLM.from_pretrained(TEACHER_PATH, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
print("[DISTILL] Teacher loaded.")

print("[DISTILL] Loading student...")
student = AutoModelForCausalLM.from_pretrained(STUDENT_PATH, device_map="auto", torch_dtype=torch.bfloat16)

lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], bias="none")
student = get_peft_model(student, lora)
student.print_trainable_parameters()

class ScubaDS(Dataset):
    def __init__(self, path, tok, max_len=1024):
        self.samples = []
        with open(path) as f:
            for line in f:
                line=line.strip()
                if not line: continue
                obj=json.loads(line)
                text=tok.apply_chat_template(obj["messages"],tokenize=False,add_generation_prompt=False)
                enc=tok(text,truncation=True,max_length=max_len,padding="max_length",return_tensors="pt")
                self.samples.append({"input_ids":enc["input_ids"].squeeze(),"attention_mask":enc["attention_mask"].squeeze(),"labels":enc["input_ids"].squeeze().clone()})
    def __len__(self): return len(self.samples)
    def __getitem__(self,i): return self.samples[i]

print("[DISTILL] Loading dataset...")
ds = ScubaDS(DATASET_PATH, tok)
print(f"[DISTILL] {len(ds)} samples.")

teacher_device = next(teacher.parameters()).device

class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        iids=inputs["input_ids"]; mask=inputs["attention_mask"]; labels=inputs["labels"]
        s_out=model(input_ids=iids,attention_mask=mask)
        s_logits=s_out.logits
        sl=s_logits[...,:-1,:].contiguous(); ll=labels[...,1:].contiguous()
        ce=F.cross_entropy(sl.view(-1,sl.size(-1)),ll.view(-1),ignore_index=tok.pad_token_id)
        with torch.no_grad():
            t_logits=teacher(input_ids=iids.to(teacher_device),attention_mask=mask.to(teacher_device)).logits.to(iids.device)
        T=TEMPERATURE
        slp=F.log_softmax(sl/T,dim=-1)
        tp=F.softmax(t_logits[...,:-1,:].contiguous()/T,dim=-1)
        m=(ll!=tok.pad_token_id).float().unsqueeze(-1)
        kl=F.kl_div(slp,tp,reduction="none")
        kl=(kl.sum(-1,keepdim=True)*m).sum()/m.sum()*(T**2)
        loss=ALPHA_KL*kl+ALPHA_CE*ce
        return (loss,s_out) if return_outputs else loss

args = TrainingArguments(
    output_dir=SFT_OUT, num_train_epochs=1,
    per_device_train_batch_size=1, gradient_accumulation_steps=16,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=10,
    bf16=True, logging_steps=5, save_steps=50, save_total_limit=2,
    optim="adamw_torch_fused", dataloader_num_workers=1,
    remove_unused_columns=False, report_to="none", gradient_checkpointing=True,
)
trainer = DistillTrainer(model=student, args=args, train_dataset=ds, processing_class=tok)
print("[DISTILL] Training started...")
trainer.train()
trainer.save_model(SFT_OUT)
tok.save_pretrained(SFT_OUT)
print(f"[DISTILL] Done -> {SFT_OUT}")
del teacher
torch.cuda.empty_cache()
PYEOF

# ── PHASE 2: DPO ──────────────────────────────────────────────
echo "============================================================"
echo " [2/4] DPO PERSONA BAKING"
echo "============================================================"
python3 << PYEOF
import os, torch, json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import PeftModel
from tqdm import tqdm

STUDENT_PATH = "/home/sesterce/sovereign_scuba_workspace/student"
SFT_OUT      = "/home/sesterce/sovereign_scuba_workspace/sft_out"
DATASET_PATH = "/home/sesterce/sovereign_scuba_workspace/scuba_bible.jsonl"
DPO_OUT      = "/home/sesterce/sovereign_scuba_workspace/dpo_out"

tok = AutoTokenizer.from_pretrained(SFT_OUT, use_fast=True)
tok.pad_token = tok.eos_token
tok.padding_side = "left"

raw_dpo = []
sft_samples = []
with open(DATASET_PATH) as f:
    for line in f:
        line=line.strip()
        if not line: continue
        obj=json.loads(line)
        if "chosen" in obj and "rejected" in obj and "prompt" in obj:
            raw_dpo.append({"prompt":obj["prompt"],"chosen":obj["chosen"],"rejected":obj["rejected"]})
        else:
            sft_samples.append(obj)

print(f"[DPO] Pre-existing pairs: {len(raw_dpo)}")
print("[DPO] Loading base student for rejection sampling...")
base = AutoModelForCausalLM.from_pretrained(STUDENT_PATH, device_map="auto", torch_dtype=torch.bfloat16)
base.eval()

for i in tqdm(range(0, min(len(sft_samples), 100), 4), desc="[DPO] Sampling"):
    batch=sft_samples[i:i+4]
    prompts,chosens=[],[]
    for obj in batch:
        u=[m["content"] for m in obj["messages"] if m["role"]=="user"]
        a=[m["content"] for m in obj["messages"] if m["role"]=="assistant"]
        if u and a: prompts.append(u[-1]); chosens.append(a[-1])
    if not prompts: continue
    try:
        fmt=[tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True) for p in prompts]
        enc=tok(fmt,return_tensors="pt",padding=True,truncation=True,max_length=512).to(base.device)
        with torch.no_grad():
            out=base.generate(**enc,max_new_tokens=150,temperature=0.3,do_sample=True,pad_token_id=tok.eos_token_id)
        for p,c,o in zip(prompts,chosens,out):
            r=tok.decode(o[enc["input_ids"].shape[1]:],skip_special_tokens=True).strip()
            if len(c)>=40 and len(r)>=20:
                raw_dpo.append({"prompt":p,"chosen":c,"rejected":r})
    except Exception as e:
        print(f"[DPO] Warning: {e}")

print(f"[DPO] Total pairs: {len(raw_dpo)}")
del base
torch.cuda.empty_cache()
tok.padding_side = "right"

print("[DPO] Loading SFT model...")
model=AutoModelForCausalLM.from_pretrained(STUDENT_PATH,device_map="auto",torch_dtype=torch.bfloat16)
model=PeftModel.from_pretrained(model,SFT_OUT,is_trainable=True)
ref=AutoModelForCausalLM.from_pretrained(STUDENT_PATH,device_map="auto",torch_dtype=torch.bfloat16)
ref=PeftModel.from_pretrained(ref,SFT_OUT,is_trainable=False)
ref.eval()

cfg=DPOConfig(
    output_dir=DPO_OUT, num_train_epochs=1,
    per_device_train_batch_size=1, gradient_accumulation_steps=16,
    learning_rate=5e-5, beta=0.1, lr_scheduler_type="cosine", warmup_steps=5,
    bf16=True, logging_steps=5, save_steps=50, save_total_limit=2,
    optim="adamw_torch_fused", report_to="none", gradient_checkpointing=True,
    remove_unused_columns=False, max_length=1024, max_prompt_length=512,
)
trainer=DPOTrainer(model=model,ref_model=ref,args=cfg,train_dataset=HFDataset.from_list(raw_dpo),processing_class=tok)
print(f"[DPO] Training on {len(raw_dpo)} pairs...")
trainer.train()
trainer.save_model(DPO_OUT)
tok.save_pretrained(DPO_OUT)
print(f"[DPO] Done -> {DPO_OUT}")
del model,ref,trainer
torch.cuda.empty_cache()
PYEOF

# ── PHASE 3: MERGE ────────────────────────────────────────────
echo "============================================================"
echo " [3/4] MERGING"
echo "============================================================"
python3 << PYEOF
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

STUDENT_PATH="/home/sesterce/sovereign_scuba_workspace/student"
DPO_OUT="/home/sesterce/sovereign_scuba_workspace/dpo_out"
SFT_OUT="/home/sesterce/sovereign_scuba_workspace/sft_out"
MERGED_OUT="/home/sesterce/sovereign_scuba_workspace/merged"

adapter=DPO_OUT if os.path.exists(os.path.join(DPO_OUT,"adapter_config.json")) else SFT_OUT
print(f"[MERGE] Using: {adapter}")
tok=AutoTokenizer.from_pretrained(adapter,use_fast=True)
base=AutoModelForCausalLM.from_pretrained(STUDENT_PATH,device_map="cpu",torch_dtype=torch.float16)
model=PeftModel.from_pretrained(base,adapter)
model=model.merge_and_unload()
model.eval()
model.save_pretrained(MERGED_OUT,safe_serialization=True)
tok.save_pretrained(MERGED_OUT)
print(f"[MERGE] Done -> {MERGED_OUT}")
PYEOF

# ── PHASE 4: QUANTIZE ─────────────────────────────────────────
echo "============================================================"
echo " [4/4] BUILD llama.cpp + QUANTIZE"
echo "============================================================"
CUDA_ARCH=$(python3 -c "import torch; cap=torch.cuda.get_device_capability(); print(cap[0]*10+cap[1])")
echo "[BUILD] CUDA arch: sm_$CUDA_ARCH"

cd ~
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi
cd ~/llama.cpp && rm -rf build && mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --target llama-quantize --target llama-cli -j $(nproc)
cd ~

python3 ~/llama.cpp/convert_hf_to_gguf.py \
    ~/sovereign_scuba_workspace/merged \
    --outfile ~/sovereign_scuba_workspace/scuba_raw.gguf \
    --outtype f16

~/llama.cpp/build/bin/llama-quantize ~/sovereign_scuba_workspace/scuba_raw.gguf ~/sovereign_scuba_workspace/SOVEREIGN_SCUBA_Q4KM.gguf Q4_K_M
~/llama.cpp/build/bin/llama-quantize ~/sovereign_scuba_workspace/scuba_raw.gguf ~/sovereign_scuba_workspace/SOVEREIGN_SCUBA_Q3KM.gguf Q3_K_M
~/llama.cpp/build/bin/llama-quantize ~/sovereign_scuba_workspace/scuba_raw.gguf ~/sovereign_scuba_workspace/SOVEREIGN_SCUBA_Q2K_TINY.gguf Q2_K
rm -f ~/sovereign_scuba_workspace/scuba_raw.gguf

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         SOVEREIGN SCUBA COMPLETE                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
ls -lh ~/sovereign_scuba_workspace/*.gguf
