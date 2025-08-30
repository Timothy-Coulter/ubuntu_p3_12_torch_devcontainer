#!/usr/bin/env bash
# Prompt for and securely store API keys/tokens for common ML services.
# - Writes environment variables to .env.local at repo root (gitignored)
# - Creates ~/.kaggle/kaggle.json with strict permissions when Kaggle creds are provided
# - Optionally logs into Hugging Face using huggingface-cli if token is provided

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
ENV_FILE="${ROOT_DIR}/.env.local"
ENV_EXAMPLE="${ROOT_DIR}/.env.example"

color() { printf "\033[%sm%s\033[0m\n" "$1" "${2:-}"; }
info()  { color "36" "➤ $*"; }
pass()  { color "32" "✔ $*"; }
warn()  { color "33" "⚠ $*"; }
fail()  { color "31" "✘ $*"; }

ensure_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    touch "$f" || {
      fail "Cannot create file: $f"
      return 1
    }
    chmod 600 "$f" || warn "Cannot set permissions on $f"
  fi
}

validate_token() {
  local token="$1"
  local service="$2"
  
  # Input sanitization
  if [[ ! "$token" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    warn "Token contains potentially dangerous characters"
    return 1
  fi
  
  case "$service" in
    "huggingface")
      if [[ ! "$token" =~ ^hf_[a-zA-Z0-9]{37}$ ]]; then
        warn "Hugging Face token format looks incorrect (should start with hf_ and be 40 chars total)"
        return 1
      fi
      ;;
    "openai")
      if [[ ! "$token" =~ ^sk-[a-zA-Z0-9]{48,}$ ]] && [[ ! "$token" =~ ^sk-proj-[a-zA-Z0-9_-]{43,}$ ]]; then
        warn "OpenAI token format looks incorrect"
        return 1
      fi
      ;;
    "wandb")
      if [[ ${#token} -lt 32 || ${#token} -gt 50 ]]; then
        warn "Weights & Biases token length looks incorrect"
        return 1
      fi
      ;;
    "anthropic")
      if [[ ! "$token" =~ ^sk-ant-[a-zA-Z0-9_-]{93,}$ ]]; then
        warn "Anthropic token format looks incorrect"
        return 1
      fi
      ;;
  esac
  
  return 0
}

# Secure temporary file handling
write_env() {
  local key="$1"
  local val="$2"
  
  # Enhanced validation
  if [[ -z "$key" || -z "$val" ]]; then
    warn "Skipping empty key or value: $key"
    return 0
  fi
  
  # Validate key format
  if [[ ! "$key" =~ ^[A-Z][A-Z0-9_]*$ ]]; then
    warn "Invalid environment variable name: $key"
    return 1
  fi
  
  # Create secure temporary file with explicit permissions
  local tmp_file
  tmp_file=$(mktemp "${ENV_FILE}.XXXXXX") || {
    fail "Cannot create secure temporary file"
    return 1
  }
  chmod 600 "$tmp_file"
  
  # Remove any existing line for key, then append
  if [[ -f "$ENV_FILE" ]]; then
    grep -v -E "^${key}=" "$ENV_FILE" 2>/dev/null > "$tmp_file" || :
  fi
  
  # Add new key-value pair with proper escaping
  printf "%s=%q\n" "$key" "$val" >> "$tmp_file"
  
  # Atomic move with permission preservation
  if mv "$tmp_file" "$ENV_FILE" 2>/dev/null; then
    chmod 600 "$ENV_FILE"
    pass "Securely stored: $key"
  else
    fail "Failed to write environment variable: $key"
    rm -f "$tmp_file" 2>/dev/null || :
    return 1
  fi
}

secure_kaggle() {
  local user="$1"
  local key="$2"
  local kd="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
  
  # Create directory
  mkdir -p "$kd" || {
    fail "Cannot create Kaggle config directory: $kd"
    return 1
  }
  
  local jf="$kd/kaggle.json"
  
  # Create JSON file with proper error handling
  cat > "$jf" <<JSON || {
    fail "Cannot write Kaggle credentials file"
    return 1
  }
{
  "username": "${user}",
  "key": "${key}"
}
JSON
  
  # Set secure permissions
  chmod 600 "$jf" || {
    fail "Cannot set secure permissions on Kaggle config"
    return 1
  }
  
  pass "Wrote Kaggle credentials to ${jf} with chmod 600"
}

hf_login() {
  local token="$1"
  
  if command -v huggingface-cli &>/dev/null; then
    # Avoid interactive TTY prompts; use token directly
    if echo "$token" | huggingface-cli login --token "$token" --add-to-git-credential 2>/dev/null; then
      pass "Hugging Face token configured with huggingface-cli"
    else
      warn "huggingface-cli login failed, but token is stored in .env.local"
    fi
  else
    warn "huggingface-cli not found; token stored in .env.local only"
    info "Install transformers to get huggingface-cli: pip install transformers"
  fi
}

prompt_var() {
  local prompt="$1"
  local varname="$2"
  local silent="${3:-false}"
  local val=""
  
  while true; do
    if [[ "$silent" == "true" ]]; then
      read -r -s -p "$prompt (leave empty to skip): " val
      echo  # newline after hidden input
    else
      read -r -p "$prompt (leave empty to skip): " val
    fi
    
    # Allow empty values (user can skip)
    break
  done
  
  printf "%s" "$val"
}

validate_token() {
  local token="$1"
  local service="$2"
  
  case "$service" in
    "huggingface")
      if [[ ! "$token" =~ ^hf_[a-zA-Z0-9]{37}$ ]]; then
        warn "Hugging Face token format looks incorrect (should start with hf_ and be 40 chars total)"
        return 1
      fi
      ;;
    "openai")
      if [[ ! "$token" =~ ^sk-[a-zA-Z0-9]{48,}$ ]]; then
        warn "OpenAI token format looks incorrect (should start with sk-)"
        return 1
      fi
      ;;
    "wandb")
      if [[ ${#token} -lt 32 ]]; then
        warn "Weights & Biases token looks too short"
        return 1
      fi
      ;;
  esac
  
  return 0
}

show_example_file() {
  if [[ -f "$ENV_EXAMPLE" ]]; then
    info "For reference, see the example file: $ENV_EXAMPLE"
  else
    warn "Example file not found: $ENV_EXAMPLE"
  fi
}

main() {
  info "This will store secrets in ${ENV_FILE} (gitignored) and provider-specific config dirs."
  
  # Show example file reference
  show_example_file
  
  # Ensure .env.local exists with secure permissions
  ensure_file "$ENV_FILE" || {
    fail "Cannot create environment file"
    exit 1
  }
  
  # Add header comment
  if [[ ! -s "$ENV_FILE" ]]; then
    cat > "$ENV_FILE" <<EOF
# Auto-generated environment file - DO NOT COMMIT TO GIT
# Generated on: $(date)
# Edit manually or re-run setup script to modify

EOF
  fi
  
  echo
  color "1;34" "=== Hugging Face ==="
  info "Get your token from: https://huggingface.co/settings/tokens"
  read -r -p "Configure Hugging Face token now? [y/N]: " hf_yes || hf_yes="N"
  hf_yes="${hf_yes:-N}"
  
  if [[ "$hf_yes" =~ ^[Yy]$ ]]; then
    HF_TOKEN="$(prompt_var "Enter HUGGING_FACE_TOKEN" "HUGGING_FACE_TOKEN" true)"
    if [[ -n "${HF_TOKEN:-}" ]]; then
      if validate_token "$HF_TOKEN" "huggingface"; then
        write_env "HUGGINGFACE_TOKEN" "$HF_TOKEN"
        write_env "HF_HOME" "${HF_HOME:-/home/vscode/.cache/huggingface}"
        write_env "TRANSFORMERS_CACHE" "${TRANSFORMERS_CACHE:-${HF_HOME:-/home/vscode/.cache/huggingface}}"
        hf_login "$HF_TOKEN"
      else
        warn "Token validation failed, but storing anyway"
        write_env "HUGGINGFACE_TOKEN" "$HF_TOKEN"
      fi
    fi
  fi
  
  echo
  color "1;34" "=== Kaggle ==="
  info "Get credentials from: https://www.kaggle.com/settings/account > API > Create New Token"
  read -r -p "Configure Kaggle credentials now? [y/N]: " kg_yes || kg_yes="N"
  kg_yes="${kg_yes:-N}"
  
  if [[ "$kg_yes" =~ ^[Yy]$ ]]; then
    KAGGLE_USERNAME="$(prompt_var "Enter KAGGLE_USERNAME" "KAGGLE_USERNAME" false)"
    KAGGLE_KEY="$(prompt_var "Enter KAGGLE_KEY (API token)" "KAGGLE_KEY" true)"
    
    if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
      write_env "KAGGLE_USERNAME" "$KAGGLE_USERNAME"
      write_env "KAGGLE_KEY" "$KAGGLE_KEY"
      write_env "KAGGLE_CONFIG_DIR" "${KAGGLE_CONFIG_DIR:-/home/vscode/.kaggle}"
      secure_kaggle "$KAGGLE_USERNAME" "$KAGGLE_KEY"
    elif [[ -n "${KAGGLE_USERNAME:-}" || -n "${KAGGLE_KEY:-}" ]]; then
      warn "Both username and API key are required for Kaggle"
    fi
  fi
  
  echo
  color "1;34" "=== Weights & Biases ==="
  info "Get API key from: https://wandb.ai/settings"
  read -r -p "Configure Weights & Biases API key now? [y/N]: " wb_yes || wb_yes="N"
  wb_yes="${wb_yes:-N}"
  
  if [[ "$wb_yes" =~ ^[Yy]$ ]]; then
    WANDB_API_KEY="$(prompt_var "Enter WANDB_API_KEY" "WANDB_API_KEY" true)"
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
      if validate_token "$WANDB_API_KEY" "wandb"; then
        write_env "WANDB_API_KEY" "$WANDB_API_KEY"
        pass "Stored WANDB_API_KEY in .env.local"
      else
        warn "Token validation failed, but storing anyway"
        write_env "WANDB_API_KEY" "$WANDB_API_KEY"
      fi
    fi
  fi
  
  echo
  color "1;34" "=== OpenAI ==="
  info "Get API key from: https://platform.openai.com/api-keys"
  read -r -p "Configure OpenAI API key now? [y/N]: " oa_yes || oa_yes="N"
  oa_yes="${oa_yes:-N}"
  
  if [[ "$oa_yes" =~ ^[Yy]$ ]]; then
    OPENAI_API_KEY="$(prompt_var "Enter OPENAI_API_KEY" "OPENAI_API_KEY" true)"
    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
      if validate_token "$OPENAI_API_KEY" "openai"; then
        write_env "OPENAI_API_KEY" "$OPENAI_API_KEY"
        pass "Stored OPENAI_API_KEY in .env.local"
      else
        warn "Token validation failed, but storing anyway"
        write_env "OPENAI_API_KEY" "$OPENAI_API_KEY"
      fi
    fi
  fi
  
  echo
  color "1;34" "=== Anthropic (Claude) ==="
  info "Get API key from: https://console.anthropic.com/"
  read -r -p "Configure Anthropic API key now? [y/N]: " ant_yes || ant_yes="N"
  ant_yes="${ant_yes:-N}"
  
  if [[ "$ant_yes" =~ ^[Yy]$ ]]; then
    ANTHROPIC_API_KEY="$(prompt_var "Enter ANTHROPIC_API_KEY" "ANTHROPIC_API_KEY" true)"
    if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
      write_env "ANTHROPIC_API_KEY" "$ANTHROPIC_API_KEY"
      pass "Stored ANTHROPIC_API_KEY in .env.local"
    fi
  fi
  
  echo
  # Summary
  pass "Secret setup complete!"
  echo
  info "Next steps:"
  echo "  1. Your secrets are stored in: ${ENV_FILE}"
  echo "  2. File permissions set to 600 (owner read/write only)"
  echo "  3. Load in Python: from dotenv import load_dotenv; load_dotenv('.env.local')"
  echo "  4. Load in shell: export \$(grep -v '^#' .env.local | xargs)"
  echo
  warn "SECURITY REMINDER:"
  echo "  - Never commit .env.local to git (it's in .gitignore)"
  echo "  - Keep your API keys secure and rotate them regularly"
  echo "  - Use environment-specific keys for different stages"
  
  # Final validation
  if [[ -f "$ENV_FILE" && -s "$ENV_FILE" ]]; then
    local key_count
    key_count=$(grep -c "^[A-Z].*=" "$ENV_FILE" 2>/dev/null || echo "0")
    pass "Configuration complete! ${key_count} API keys configured."
  else
    warn "No configuration was saved"
  fi
}

# Trap for cleanup
trap 'rm -f "${ENV_FILE}.tmp" 2>/dev/null || true' EXIT

if [[ "${BASH_SOURCE[0]:-}" == "$0" ]]; then
  main "$@"
fi