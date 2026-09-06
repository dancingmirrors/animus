PREFIX ?= $(HOME)/.local
BINDIR = $(PREFIX)/bin
LIBDIR = $(PREFIX)/lib/animus
DESKTOPDIR = $(PREFIX)/share/applications

LEGACY_LIBDIR = $(PREFIX)/share/animus

PYTHON = python3
VENV_DIR = $(LIBDIR)/venv

TORCH_PREBUILT ?=
TORCH_PROFILE ?=
MARCH ?=
TORCH_EXTRA_OPTIONS ?=
NCNN ?=
NCNN_EXTRA_OPTIONS ?=

BUILD_ENV = PYTHON="$(PYTHON)" \
	$(if $(TORCH_PROFILE),TORCH_PROFILE="$(TORCH_PROFILE)",) \
	$(if $(MARCH),MARCH="$(MARCH)",) \
	$(if $(TORCH_EXTRA_OPTIONS),TORCH_EXTRA_OPTIONS="$(TORCH_EXTRA_OPTIONS)",)

NCNN_ENV = PYTHON="$(VENV_DIR)/bin/python" \
	$(if $(NCNN_EXTRA_OPTIONS),NCNN_EXTRA_OPTIONS="$(NCNN_EXTRA_OPTIONS)",)

.PHONY: all install uninstall clean help torch torch-ensure ncnn \
	self-test prune-legacy-install

all: help

help:
	@echo "  make install     Build torch from source and install to $(PREFIX)"
	@echo "                   (TORCH_PREBUILT=1 make install uses a prebuilt wheel)"
	@echo "  make torch       Build a CPU torch wheel from source into wheels/"
	@echo "  make ncnn        Build ncnn with Vulkan, for GPU upscaling"
	@echo "  make self-test   Check the installed upscaler against this torch"
	@echo "  make uninstall   Remove from $(PREFIX)"
	@echo "  make clean       Clean the virtual environment"
	@echo ""
	@echo "  TORCH_PROFILE=modern|legacy|auto  which torch build options to use"
	@echo "  MARCH=...        -march for the torch build (default: native)"
	@echo "  TORCH_EXTRA_OPTIONS=\"USE_FBGEMM=1 ...\"  extra torch build flags"
	@echo "  NCNN=1           also build NCNN, so the GPU shows up as a device"

prune-legacy-install:
	@rm -fr $(LEGACY_LIBDIR)/venv
	@rm -f $(LEGACY_LIBDIR)/animus.py $(LEGACY_LIBDIR)/upscale.py
	@rm -f $(LEGACY_LIBDIR)/requirements.txt
	@rm -f $(LEGACY_LIBDIR)/README.md $(LEGACY_LIBDIR)/COPYING
	@rmdir $(LEGACY_LIBDIR) 2>/dev/null || true

install: torch-ensure prune-legacy-install
	@mkdir -p $(LIBDIR)
	@mkdir -p $(BINDIR)
	@mkdir -p $(DESKTOPDIR)

	@install -m 755 animus.py $(LIBDIR)/animus.py
	@install -m 755 upscale.py $(LIBDIR)/upscale.py
	@install -m 644 requirements.txt $(LIBDIR)/requirements.txt
	@install -m 644 README.md $(LIBDIR)/README.md
	@install -m 644 COPYING $(LIBDIR)/COPYING

	@if [ -d "$(VENV_DIR)" ] && [ -n "$(VENV_DIR)" ] && echo "$(VENV_DIR)" | grep -q "lib/animus/venv"; then \
		rm -fr $(VENV_DIR); \
	fi
	@$(PYTHON) -m venv $(VENV_DIR) || { \
		echo "==> This Python has no ensurepip. Falling back to virtualenv."; \
		$(PYTHON) -m virtualenv $(VENV_DIR); \
	}

	@$(VENV_DIR)/bin/pip install --upgrade pip

	@if ls wheels/torch-*.whl >/dev/null 2>&1; then \
		$(VENV_DIR)/bin/pip install wheels/torch-*.whl; \
	else \
		$(VENV_DIR)/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch; \
	fi

	@$(VENV_DIR)/bin/pip install -r requirements.txt

	@if [ "$(NCNN)" = "1" ]; then \
		$(NCNN_ENV) ./build-ncnn.sh && \
		site="$$($(VENV_DIR)/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" && \
		install -m 755 wheels/ncnn*.so "$$site" && \
		echo "==> Installed $$(basename wheels/ncnn*.so) into $$site."; \
	fi

	@install -m 755 animus $(BINDIR)/animus
	@install -m 755 animus $(BINDIR)/animus-upscale

	@sed 's|@BINDIR@|$(BINDIR)|g' animus.desktop.in > $(DESKTOPDIR)/animus.desktop
	@chmod 644 $(DESKTOPDIR)/animus.desktop
	@sed 's|@BINDIR@|$(BINDIR)|g' animus-upscale.desktop.in \
		> $(DESKTOPDIR)/animus-upscale.desktop
	@chmod 644 $(DESKTOPDIR)/animus-upscale.desktop

	@echo "==> Installed to $(PREFIX). Run 'animus' or 'animus-upscale'."

torch:
	@$(BUILD_ENV) ./build-torch.sh

ncnn:
	@$(NCNN_ENV) ./build-ncnn.sh

self-test:
	@$(VENV_DIR)/bin/python $(LIBDIR)/upscale.py --self-test

torch-ensure:
	@if [ "$(TORCH_PREBUILT)" = "1" ]; then \
		echo "==> TORCH_PREBUILT=1: will install a prebuilt CPU PyTorch."; \
	else \
		$(BUILD_ENV) ./build-torch.sh; \
	fi

uninstall: prune-legacy-install
	@rm -f $(BINDIR)/animus
	@rm -f $(BINDIR)/animus-upscale
	@rm -f $(DESKTOPDIR)/animus.desktop
	@rm -f $(DESKTOPDIR)/animus-upscale.desktop
	@rm -fr $(LIBDIR)

clean:
	@rm -fr venv
	@find . -name __pycache__ -type d -prune -exec rm -fr {} + 2>/dev/null || true
	@find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@find . -type f -name '*.pyo' -delete 2>/dev/null || true
