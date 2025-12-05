# Model Setup Instructions

## Issue: Missing Model Weight Files

The model directory (`models/model/pytorch_modell/`) exists but is missing the actual model weight files. The index file (`model.safetensors.index.json`) references 4 shard files that need to be present:

- `model-00001-of-00004.safetensors`
- `model-00002-of-00004.safetensors`
- `model-00003-of-00004.safetensors`
- `model-00004-of-00004.safetensors`

## Solution Options

### Option 1: Download Model from HuggingFace (Requires Authentication)

1. **Request access to Llama 3.1 8B Instruct**:
   - Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
   - Request access (requires HuggingFace account)
   - Wait for approval

2. **Authenticate with HuggingFace**:
   ```bash
   huggingface-cli login
   ```
   Enter your HuggingFace token when prompted.

3. **Download the model**:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   
   model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
   model = AutoModelForCausalLM.from_pretrained(
       model_id,
       torch_dtype=torch.float16
   )
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   
   # Save locally
   model.save_pretrained("models/model/pytorch_modell")
   tokenizer.save_pretrained("models/model/Tokenizer")
   ```

### Option 2: Use an Alternative Open Model

If you don't have access to Llama, you can use an alternative open model. Update `config/rag_system_config.json`:

```json
{
  "model": {
    "model_id": "microsoft/Phi-3-mini-4k-instruct",
    "model": null
  }
}
```

Or use other open models like:
- `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters)
- `mistralai/Mistral-7B-Instruct-v0.2` (requires access)
- `google/gemma-2b-it` (smaller, open)

### Option 3: Use a Smaller Local Model

If you have a different model saved elsewhere, update the paths in `config/rag_system_config.json` to point to that location.

## Verification

After setting up the model, verify it works:

```bash
python test_load.py
```

This should load the model successfully without errors.

