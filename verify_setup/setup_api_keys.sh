#!/usr/bin/env bash
# Prompt for and securely store API keys/tokens for common ML services.
# - Writes environment variables to .env.local at repo root (gitignored)
# - Creates ~/.kaggle/kaggle.json with strict permissions when Kaggle creds are provided
# - Optionally logs into Hugging Face using huggingface-cli if token is provided

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
ENV_FILE="${ROOT_DIR}/.env.local"

color() { printf "\033[%sm%s\033[0m\n" "$1" "${2:-}"; }
info()  { color "36" "➤ $*"; }
pass()  { color "32" "✔ $*"; }
warn()  { color "33" "⚠ $*"; }
fail()  { color "31" "✘ $*"; }

ensure_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    touch "$f"
  fi
}

write_env() {
  local key="$1"
  local val="$2"
  # Remove any existing line for key, then append
  grep -v -E "^${key}=" "$ENV_FILE" 2>/dev/null > "${ENV_FILE}.tmp" || true
  mv "${ENV_FILE}.tmp" "$ENV_FILE"
  printf "%s=%s\n" "$key" "$val" >> "$ENV_FILE"
}

secure_kaggle() {
  local user="$1"
  local key="$2"
  local kd="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
  mkdir -p "$kd"
  local jf="$kd/kaggle.json"
  cat > "$jf" <<JSON
{
  "username": "${user}",
  "key": "${key}"
}
JSON
  chmod 600 "$jf"
  pass "Wrote Kaggle credentials to ${jf} with chmod 600"
}

hf_login() {
  local token="$1"
  if command -v huggingface-cli &>/dev/null; then
    # Avoid interactive TTY prompts; use token directly
    huggingface-cli login --token "$token" --add-to-git-credential || warn "huggingface-cli login returned non-zero"
    pass "Hugging Face token configured with huggingface-cli"
  else
    warn "huggingface-cli not found; token stored in .env.local only"
  fi
}

prompt_var() {
  local prompt="$1"
  local varname="$2"
  local silent="${3:-false}"
  local val=""
  if [[ "$silent" == "true" ]]; then
    read -r -s -p "$prompt: " val; echo
  else
    read -r -p "$prompt: " val
  fi
  printf "%s" "$val"
}

main() {
  info "This will store secrets in ${ENV_FILE} (gitignored) and provider-specific config dirs."

  ensure_file "$ENV_FILE"

  echo
  color "1;34" "=== Hugging Face ==="
  read -r -p "Configure Hugging Face token now? [y/N]: " hf_yes || true
  hf_yes="${hf_yes:-N}"
  if [[ "$hf_yes" =~ ^[Yy]$ ]]; then
    HF_TOKEN="$(prompt_var "Enter HUGGING_FACE_TOKEN" "HUGGING_FACE_TOKEN" true)"
    if [[ -n "${HF_TOKEN:-}" ]]; then
      write_env "HUGGINGFACE_TOKEN" "$HF_TOKEN"
      write_env "HF_HOME" "${HF_HOME:-$HOME/.cache/huggingface}"
      write_env "TRANSFORMERS_CACHE" "${TRANSFORMERS_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
      hf_login "$HF_TOKEN"
    fi
  fi

  echo
  color "1;34" "=== Kaggle ==="
  read -r -p "Configure Kaggle credentials now? [y/N]: " kg_yes || true
  kg_yes="${kg_yes:-N}"
  if [[ "$kg_yes" =~ ^[Yy]$ ]]; then
    KAGGLE_USERNAME="$(prompt_var "Enter KAGGLE_USERNAME" "KAGGLE_USERNAME" false)"
    KAGGLE_KEY="$(prompt_var "Enter KAGGLE_KEY (API token)" "KAGGLE_KEY" true)"
    if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
      write_env "KAGGLE_USERNAME" "$KAGGLE_USERNAME"
      write_env "KAGGLE_KEY" "$KAGGLE_KEY"
      write_env "KAGGLE_CONFIG_DIR" "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
      secure_kaggle "$KAGGLE_USERNAME" "$KAGGLE_KEY"
    fi
  fi

  echo
  color "1;34" "=== Optional: Weights & Biases ==="
  read -r -p "Configure Weights & Biases API key now? [y/N]: " wb_yes || true
  wb_yes="${wb_yes:-N}"
  if [[ "$wb_yes" =~ ^[Yy]$ ]]; then
    WANDB_API_KEY="$(prompt_var "Enter WANDB_API_KEY" "WANDB_API_KEY" true)"
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
      write_env "WANDB_API_KEY" "$WANDB_API_KEY"
      pass "Stored WANDB_API_KEY in .env.local"
    fi
  fi

  echo
  color "1;34" "=== Optional: OpenAI ==="
  read -r -p "Configure OpenAI API key now? [y/N]: " oa_yes || true
  oa_yes="${oa_yes:-N}"
  if [[ "$oa_yes" =~ ^[Yy]$ ]]; then
    OPENAI_API_KEY="$(prompt_var "Enter OPENAI_API_KEY" "OPENAI_API_KEY" true)"
    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
      write_env "OPENAI_API_KEY" "$OPENAI_API_KEY"
      pass "Stored OPENAI_API_KEY in .env.local"
    fi
  fi

  echo
  pass "Secret setup complete. To load these in a shell: export \$(grep -v '^#' .env.local | xargs) (or use direnv)."
  echo "Note: VS Code devcontainer does not automatically load .env.local into container env; use dotenv in your app or source it in your shell."
}

if [[ "${BASH_SOURCE[0]:-}" == "$0" ]]; then
  main "$@"
fi