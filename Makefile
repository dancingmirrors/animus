PREFIX ?= $(HOME)/.local
BINDIR = $(PREFIX)/bin
SHAREDIR = $(PREFIX)/share/animus
DESKTOPDIR = $(PREFIX)/share/applications

PYTHON = python3
VENV_DIR = $(SHAREDIR)/venv

TORCH_PREBUILT ?=

.PHONY: all install uninstall clean help torch torch-ensure

all: help

help:
	@echo "  make install     Build torch from source and install to $(PREFIX)"
	@echo "                   (TORCH_PREBUILT=1 make install uses a prebuilt wheel)"
	@echo "  make torch       Build a CPU torch wheel from source into wheels/"
	@echo "  make uninstall   Remove from $(PREFIX)"
	@echo "  make clean       Clean the virtual environment"

install: torch-ensure
	@mkdir -p $(SHAREDIR)
	@mkdir -p $(BINDIR)
	@mkdir -p $(DESKTOPDIR)

	@install -m 755 animus.py $(SHAREDIR)/animus.py
	@install -m 644 requirements.txt $(SHAREDIR)/requirements.txt
	@install -m 644 README.md $(SHAREDIR)/README.md
	@install -m 644 COPYING $(SHAREDIR)/COPYING

	@if [ -d "$(VENV_DIR)" ] && [ -n "$(VENV_DIR)" ] && echo "$(VENV_DIR)" | grep -q "share/animus/venv"; then \
		rm -rf $(VENV_DIR); \
	fi
	@$(PYTHON) -m venv $(VENV_DIR)

	@$(VENV_DIR)/bin/pip install --upgrade pip

	@if ls wheels/torch-*.whl >/dev/null 2>&1; then \
		$(VENV_DIR)/bin/pip install wheels/torch-*.whl; \
	else \
		$(VENV_DIR)/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch; \
	fi

	@$(VENV_DIR)/bin/pip install -r requirements.txt
	@install -m 755 animus $(BINDIR)/animus

	@sed 's|@BINDIR@|$(BINDIR)|g' animus.desktop.in > $(DESKTOPDIR)/animus.desktop
	@chmod 644 $(DESKTOPDIR)/animus.desktop

torch:
	@PYTHON="$(PYTHON)" ./build-torch.sh

torch-ensure:
	@if [ "$(TORCH_PREBUILT)" = "1" ]; then \
		echo "==> TORCH_PREBUILT=1: will install a prebuilt CPU PyTorch."; \
	elif [ -d .torch-src/.git ]; then \
		echo "==> An incremental build of .torch-src..."; \
		PYTHON="$(PYTHON)" ./build-torch.sh; \
	elif ls wheels/torch-*.whl >/dev/null 2>&1; then \
		echo "==> Reusing the existing wheels..."; \
	else \
		echo "==> Building CPU PyTorch from source."; \
		PYTHON="$(PYTHON)" ./build-torch.sh; \
	fi

uninstall:
	@rm -f $(BINDIR)/animus
	@rm -f $(DESKTOPDIR)/animus.desktop
	@rm -rf $(SHAREDIR)

clean:
	@rm -rf venv
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@find . -type f -name '*.pyo' -delete 2>/dev/null || true
