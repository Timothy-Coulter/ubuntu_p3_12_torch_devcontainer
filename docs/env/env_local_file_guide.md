# Environment Configuration Setup

## 📍 **File Location**

Create `.env.local` in your **project root directory**:

```
your-project/
├── .devcontainer/
├── .env.local          ← HERE (same level as pyproject.toml)
├── pyproject.toml
└── ...
```

## 🔧 **Setup Process**

### 1. Create the file
```bash
# In your project root
cp templates/env.local .env.local
```

### 2. Edit with your credentials
```bash
# Use any editor
nano .env.local
# or
code .env.local
```

### 3. Verify it's ignored by git
```bash
git status  # .env.local should NOT appear in untracked files
```

## 🔒 **Security Best Practices**

1. **Never commit `.env.local`** (it's in `.gitignore`)
2. **Set restrictive permissions**: `chmod 600 .env.local`
3. **Use different files for different environments**:
   - `.env.local` - Development
   - `.env.staging` - Staging
   - `.env.production` - Production

## 🏗️ **Container Access**

The container reads `.env.local` in several ways:

### Automatic Loading (Recommended)
```python
from dotenv import load_dotenv
load_dotenv('.env.local')  # Loads from project root
```

### Manual Export (Alternative)
```bash
# In container terminal
export $(grep -v '^#' .env.local | xargs)
```

### VS Code Integration
VS Code dev containers automatically make the file available at the workspace folder level.

## 🎯 **Interactive Setup**

Use the interactive setup script:
```bash
./dev.sh setup-keys
```

This script will:
- Create `.env.local` with secure permissions
- Prompt for each API key
- Validate token formats
- Set up service-specific configs (like Kaggle JSON)

## 📂 **File Structure Example**

```
# Your actual .env.local file
HUGGINGFACE_TOKEN=hf_your_actual_token_here
WANDB_API_KEY=your_actual_wandb_key
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
OPENAI_API_KEY=sk-your_actual_openai_key
```

## ⚠️ **Troubleshooting**

### File not found in container
1. Ensure file is in project root (not in `.devcontainer/`)
2. Rebuild container if file was added after initial build
3. Check file permissions: `ls -la .env.local`

### Variables not loading
1. Verify file format (KEY=value, no spaces around =)
2. Check for hidden characters or BOM
3. Test manual loading: `python -c "from dotenv import load_dotenv; load_dotenv('.env.local'); import os; print(os.getenv('YOUR_KEY'))"`