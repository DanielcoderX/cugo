package tensor

import (
	"errors"
	"math"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/kernels/optim"
)

// Optimizer defines the standard gradient descent optimization interface.
type Optimizer interface {
	Step() error
	ZeroGrad()
}

// -----------------------------------------------------------------------------
// AdamW Optimizer
// -----------------------------------------------------------------------------

// AdamW implements the AdamW optimizer with decoupled weight decay (Loshchilov & Hutter, 2019).
type AdamW struct {
	params      []*Tensor
	expAvgM     []*Tensor
	expAvgV     []*Tensor
	lr          float32
	beta1       float32
	beta2       float32
	eps         float32
	weightDecay float32
	stepCount   int
	ctx         *driver.Context
}

// NewAdamW creates a new AdamW optimizer for the given trainable parameters.
func NewAdamW(params []*Tensor, lr, beta1, beta2, eps, weightDecay float32) (*AdamW, error) {
	if len(params) == 0 {
		return nil, errors.New("cugo/optim: params slice cannot be empty")
	}
	ctx := params[0].ctx

	expAvgM := make([]*Tensor, len(params))
	expAvgV := make([]*Tensor, len(params))

	for i, p := range params {
		zeros := make([]float32, p.Numel())
		m, err := NewFromFloat32(ctx, zeros, p.shape...)
		if err != nil {
			return nil, err
		}
		v, err := NewFromFloat32(ctx, zeros, p.shape...)
		if err != nil {
			_ = m.Close()
			return nil, err
		}
		expAvgM[i] = m
		expAvgV[i] = v
	}

	return &AdamW{
		params:      params,
		expAvgM:     expAvgM,
		expAvgV:     expAvgV,
		lr:          lr,
		beta1:       beta1,
		beta2:       beta2,
		eps:         eps,
		weightDecay: weightDecay,
		stepCount:   0,
		ctx:         ctx,
	}, nil
}

// Step performs a single parameter update step across all registered tensors.
func (opt *AdamW) Step() error {
	opt.stepCount++

	bc1 := float32(1.0 - math.Pow(float64(opt.beta1), float64(opt.stepCount)))
	bc2 := float32(1.0 - math.Pow(float64(opt.beta2), float64(opt.stepCount)))

	mod, err := opt.ctx.LoadModuleData(optim.PTX)
	if err != nil {
		return err
	}
	defer mod.Unload()

	fn, err := mod.Function("adamwKernel")
	if err != nil {
		return err
	}

	for i, p := range opt.params {
		if p.Grad == nil {
			continue
		}

		n := int32(p.Numel())
		cfg := driver.LaunchConfig{
			GridDimX:  uint32((n + 255) / 256),
			BlockDimX: 256,
		}

		if err := fn.Launch(
			cfg,
			p.DevicePtr(),
			p.Grad.DevicePtr(),
			opt.expAvgM[i].DevicePtr(),
			opt.expAvgV[i].DevicePtr(),
			opt.lr,
			opt.beta1,
			opt.beta2,
			opt.eps,
			opt.weightDecay,
			bc1,
			bc2,
			n,
		); err != nil {
			return err
		}
	}

	return nil
}

// ZeroGrad resets all gradients on registered parameters.
func (opt *AdamW) ZeroGrad() {
	for _, p := range opt.params {
		p.ZeroGrad()
	}
}

// Close frees AdamW moment state buffers.
func (opt *AdamW) Close() {
	for _, m := range opt.expAvgM {
		_ = m.Close()
	}
	for _, v := range opt.expAvgV {
		_ = v.Close()
	}
}

// -----------------------------------------------------------------------------
// SGD Optimizer
// -----------------------------------------------------------------------------

// SGD implements stochastic gradient descent with momentum and weight decay.
type SGD struct {
	params      []*Tensor
	velocity    []*Tensor
	lr          float32
	momentum    float32
	weightDecay float32
	ctx         *driver.Context
}

// NewSGD creates a new SGD optimizer for the given trainable parameters.
func NewSGD(params []*Tensor, lr, momentum, weightDecay float32) (*SGD, error) {
	if len(params) == 0 {
		return nil, errors.New("cugo/optim: params slice cannot be empty")
	}
	ctx := params[0].ctx

	var vel []*Tensor
	if momentum != 0.0 {
		vel = make([]*Tensor, len(params))
		for i, p := range params {
			zeros := make([]float32, p.Numel())
			v, err := NewFromFloat32(ctx, zeros, p.shape...)
			if err != nil {
				return nil, err
			}
			vel[i] = v
		}
	}

	return &SGD{
		params:      params,
		velocity:    vel,
		lr:          lr,
		momentum:    momentum,
		weightDecay: weightDecay,
		ctx:         ctx,
	}, nil
}

// Step performs a parameter update step.
func (opt *SGD) Step() error {
	mod, err := opt.ctx.LoadModuleData(optim.PTX)
	if err != nil {
		return err
	}
	defer mod.Unload()

	fn, err := mod.Function("sgdKernel")
	if err != nil {
		return err
	}

	for i, p := range opt.params {
		if p.Grad == nil {
			continue
		}

		n := int32(p.Numel())
		cfg := driver.LaunchConfig{
			GridDimX:  uint32((n + 255) / 256),
			BlockDimX: 256,
		}

		var velPtr driver.DevicePtr
		if len(opt.velocity) > i && opt.velocity[i] != nil {
			velPtr = opt.velocity[i].DevicePtr()
		}

		if err := fn.Launch(
			cfg,
			p.DevicePtr(),
			p.Grad.DevicePtr(),
			velPtr,
			opt.lr,
			opt.momentum,
			opt.weightDecay,
			n,
		); err != nil {
			return err
		}
	}

	return nil
}

// ZeroGrad resets all gradients on registered parameters.
func (opt *SGD) ZeroGrad() {
	for _, p := range opt.params {
		p.ZeroGrad()
	}
}

// Close frees SGD velocity buffers.
func (opt *SGD) Close() {
	for _, v := range opt.velocity {
		if v != nil {
			_ = v.Close()
		}
	}
}
