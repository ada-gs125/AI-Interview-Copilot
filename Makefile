.PHONY: dev backend frontend test install db db-down

dev backend frontend test install db db-down:
	$(MAKE) -f dev/Makefile.dev $@
