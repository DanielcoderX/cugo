package tensor

import (
	"sync"

	"github.com/cugo/cugo/driver"
)

type moduleKey struct {
	ctxHandle uintptr
	tag       string
}

var (
	modMu    sync.Mutex
	modCache = make(map[moduleKey]*driver.Module)
)

func getModule(ctx *driver.Context, tag string, ptx []byte) (*driver.Module, error) {
	modMu.Lock()
	defer modMu.Unlock()

	key := moduleKey{ctxHandle: ctx.Handle(), tag: tag}
	if m, ok := modCache[key]; ok {
		return m, nil
	}

	m, err := ctx.LoadModuleData(ptx)
	if err != nil {
		return nil, err
	}
	modCache[key] = m
	return m, nil
}
