package tensor

import (
	"errors"
	"fmt"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/kernels/activation"
	"github.com/cugo/cugo/kernels/autograd"
	"github.com/cugo/cugo/kernels/gemm"
)

// transposeMatrix allocates and computes a 2D matrix transpose on the GPU.
func transposeMatrix(ctx *driver.Context, in *Tensor) (*Tensor, error) {
	if in.Dims() != 2 {
		return nil, errors.New("cugo/ops: transposeMatrix requires 2D tensor")
	}
	rows := int32(in.shape[0])
	cols := int32(in.shape[1])

	out, err := New(ctx, Float32, int64(cols), int64(rows))
	if err != nil {
		return nil, err
	}

	mod, err := getModule(ctx, "autograd", autograd.PTX)
	if err != nil {
		_ = out.Close()
		return nil, err
	}

	fn, err := mod.Function("transposeKernel")
	if err != nil {
		_ = out.Close()
		return nil, err
	}

	cfg := driver.LaunchConfig{
		GridDimX:  uint32((cols + 15) / 16),
		GridDimY:  uint32((rows + 15) / 16),
		BlockDimX: 16,
		BlockDimY: 16,
	}

	if err := fn.Launch(cfg, in.DevicePtr(), out.DevicePtr(), rows, cols); err != nil {
		_ = out.Close()
		return nil, err
	}
	return out, nil
}

// executeGEMM executes C = A * B on the GPU where A is [M, K], B is [K, N], C is [M, N].
func executeGEMM(ctx *driver.Context, A, B, C *Tensor, M, N, K int32) error {
	mod, err := getModule(ctx, "gemm", gemm.PTX)
	if err != nil {
		return err
	}

	fn, err := mod.Function("gemm")
	if err != nil {
		return err
	}

	cfg := driver.LaunchConfig{
		GridDimX:  uint32((N + 15) / 16),
		GridDimY:  uint32((M + 15) / 16),
		BlockDimX: 16,
		BlockDimY: 16,
	}

	return fn.Launch(cfg, A.DevicePtr(), B.DevicePtr(), C.DevicePtr(), M, N, K)
}

// -----------------------------------------------------------------------------
// MatMul Node
// -----------------------------------------------------------------------------

type matMulNode struct {
	out  *Tensor
	a, b *Tensor
}

func (n *matMulNode) Inputs() []*Tensor {
	return []*Tensor{n.a, n.b}
}

func (n *matMulNode) Backward() error {
	if n.out == nil || n.out.Grad == nil {
		return nil
	}
	gradOut := n.out.Grad

	M := int32(n.a.shape[0])
	K := int32(n.a.shape[1])
	N := int32(n.b.shape[1])
	ctx := n.a.ctx

	// 1. dA = gradOut * B^T -> [M, N] * [N, K] = [M, K]
	if n.a.RequiresGrad {
		bT, err := transposeMatrix(ctx, n.b)
		if err != nil {
			return err
		}
		defer bT.Close()

		dA, err := New(ctx, Float32, int64(M), int64(K))
		if err != nil {
			return err
		}
		defer dA.Close()

		if err := executeGEMM(ctx, gradOut, bT, dA, M, K, N); err != nil {
			return err
		}
		if err := n.a.AccumulateGrad(dA); err != nil {
			return err
		}
	}

	// 2. dB = A^T * gradOut -> [K, M] * [M, N] = [K, N]
	if n.b.RequiresGrad {
		aT, err := transposeMatrix(ctx, n.a)
		if err != nil {
			return err
		}
		defer aT.Close()

		dB, err := New(ctx, Float32, int64(K), int64(N))
		if err != nil {
			return err
		}
		defer dB.Close()

		if err := executeGEMM(ctx, aT, gradOut, dB, K, N, M); err != nil {
			return err
		}
		if err := n.b.AccumulateGrad(dB); err != nil {
			return err
		}
	}

	return nil
}

// MatMul executes matrix multiplication with autograd tracking.
func MatMul(a, b *Tensor) (*Tensor, error) {
	if a.Dims() != 2 || b.Dims() != 2 || a.shape[1] != b.shape[0] {
		return nil, fmt.Errorf("cugo/ops: MatMul incompatible shapes %v and %v", a.shape, b.shape)
	}

	M := int32(a.shape[0])
	K := int32(a.shape[1])
	N := int32(b.shape[1])

	out, err := New(a.ctx, Float32, int64(M), int64(N))
	if err != nil {
		return nil, err
	}

	if err := executeGEMM(a.ctx, a, b, out, M, N, K); err != nil {
		_ = out.Close()
		return nil, err
	}

	if a.RequiresGrad || b.RequiresGrad {
		out.RequiresGrad = true
		out.GradFn = &matMulNode{out: out, a: a, b: b}
	}

	return out, nil
}

// -----------------------------------------------------------------------------
// AddBias Node
// -----------------------------------------------------------------------------

type addBiasNode struct {
	out     *Tensor
	x, bias *Tensor
}

func (n *addBiasNode) Inputs() []*Tensor {
	return []*Tensor{n.x, n.bias}
}

func (n *addBiasNode) Backward() error {
	if n.out == nil || n.out.Grad == nil {
		return nil
	}
	gradOut := n.out.Grad
	ctx := n.x.ctx

	if n.x.RequiresGrad {
		if err := n.x.AccumulateGrad(gradOut); err != nil {
			return err
		}
	}

	if n.bias.RequiresGrad {
		rows := int32(n.x.shape[0])
		cols := int32(n.x.shape[1])

		dBias, err := New(ctx, Float32, int64(cols))
		if err != nil {
			return err
		}
		defer dBias.Close()

		mod, err := getModule(ctx, "autograd", autograd.PTX)
		if err != nil {
			return err
		}

		fn, err := mod.Function("biasBackwardKernel")
		if err != nil {
			return err
		}

		cfg := driver.LaunchConfig{
			GridDimX:  uint32((cols + 255) / 256),
			BlockDimX: 256,
		}

		if err := fn.Launch(cfg, gradOut.DevicePtr(), dBias.DevicePtr(), rows, cols); err != nil {
			return err
		}
		if err := n.bias.AccumulateGrad(dBias); err != nil {
			return err
		}
	}

	return nil
}

// AddBias adds 1D bias vector to 2D matrix rows with autograd tracking.
func AddBias(x, bias *Tensor) (*Tensor, error) {
	if x.Dims() != 2 || bias.Dims() != 1 || x.shape[1] != bias.shape[0] {
		return nil, fmt.Errorf("cugo/ops: AddBias incompatible shapes %v and %v", x.shape, bias.shape)
	}

	rows := int32(x.shape[0])
	cols := int32(x.shape[1])

	out, err := New(x.ctx, Float32, int64(rows), int64(cols))
	if err != nil {
		return nil, err
	}

	mod, err := getModule(x.ctx, "activation", activation.PTX)
	if err != nil {
		_ = out.Close()
		return nil, err
	}

	fn, err := mod.Function("biasAddKernel")
	if err != nil {
		_ = out.Close()
		return nil, err
	}

	cfg := driver.LaunchConfig{
		GridDimX:  uint32((cols + 15) / 16),
		GridDimY:  uint32((rows + 15) / 16),
		BlockDimX: 16,
		BlockDimY: 16,
	}

	if err := fn.Launch(cfg, x.DevicePtr(), bias.DevicePtr(), out.DevicePtr(), rows, cols); err != nil {
		_ = out.Close()
		return nil, err
	}

	if x.RequiresGrad || bias.RequiresGrad {
		out.RequiresGrad = true
		out.GradFn = &addBiasNode{out: out, x: x, bias: bias}
	}

	return out, nil
}

// -----------------------------------------------------------------------------
// GELU Node
// -----------------------------------------------------------------------------

type geluNode struct {
	out *Tensor
	x   *Tensor
}

func (n *geluNode) Inputs() []*Tensor {
	return []*Tensor{n.x}
}

func (n *geluNode) Backward() error {
	if n.out == nil || n.out.Grad == nil || !n.x.RequiresGrad {
		return nil
	}
	gradOut := n.out.Grad

	total := int32(n.x.Numel())
	ctx := n.x.ctx

	dX, err := New(ctx, Float32, n.x.shape...)
	if err != nil {
		return err
	}
	defer dX.Close()

	mod, err := getModule(ctx, "autograd", autograd.PTX)
	if err != nil {
		return err
	}

	fn, err := mod.Function("geluBackwardKernel")
	if err != nil {
		return err
	}

	cfg := driver.LaunchConfig{
		GridDimX:  uint32((total + 255) / 256),
		BlockDimX: 256,
	}

	if err := fn.Launch(cfg, gradOut.DevicePtr(), n.x.DevicePtr(), dX.DevicePtr(), total); err != nil {
		return err
	}

	return n.x.AccumulateGrad(dX)
}

// GELU executes Gaussian Error Linear Unit activation with autograd tracking.
func GELU(x *Tensor) (*Tensor, error) {
	out, err := New(x.ctx, Float32, x.shape...)
	if err != nil {
		return nil, err
	}

	total := int32(x.Numel())
	mod, err := getModule(x.ctx, "activation", activation.PTX)
	if err != nil {
		_ = out.Close()
		return nil, err
	}

	fn, err := mod.Function("geluKernel")
	if err != nil {
		_ = out.Close()
		return nil, err
	}

	cfg := driver.LaunchConfig{
		GridDimX:  uint32((total + 255) / 256),
		BlockDimX: 256,
	}

	if err := fn.Launch(cfg, x.DevicePtr(), out.DevicePtr(), total); err != nil {
		_ = out.Close()
		return nil, err
	}

	if x.RequiresGrad {
		out.RequiresGrad = true
		out.GradFn = &geluNode{out: out, x: x}
	}

	return out, nil
}

// -----------------------------------------------------------------------------
// MSE Loss Node
// -----------------------------------------------------------------------------

type mseLossNode struct {
	out          *Tensor
	pred, target *Tensor
}

func (n *mseLossNode) Inputs() []*Tensor {
	return []*Tensor{n.pred}
}

func (n *mseLossNode) Backward() error {
	if !n.pred.RequiresGrad {
		return nil
	}

	// d(MSE)/d(pred) = (2 / N) * (pred - target)
	numel := n.pred.Numel()
	scale := float32(2.0 / float64(numel))

	hPred, err := n.pred.ToCPUFloat32()
	if err != nil {
		return err
	}
	hTarget, err := n.target.ToCPUFloat32()
	if err != nil {
		return err
	}

	dPredHost := make([]float32, numel)
	for i := range dPredHost {
		dPredHost[i] = scale * (hPred[i] - hTarget[i])
	}

	dPred, err := NewFromFloat32(n.pred.ctx, dPredHost, n.pred.shape...)
	if err != nil {
		return err
	}
	defer dPred.Close()

	return n.pred.AccumulateGrad(dPred)
}

// MSELoss computes mean squared error between prediction and target with autograd tracking.
func MSELoss(pred, target *Tensor) (*Tensor, error) {
	if pred.Numel() != target.Numel() {
		return nil, errors.New("cugo/ops: MSELoss element count mismatch")
	}

	hPred, err := pred.ToCPUFloat32()
	if err != nil {
		return nil, err
	}
	hTarget, err := target.ToCPUFloat32()
	if err != nil {
		return nil, err
	}

	var sumSq float64
	for i := range hPred {
		diff := float64(hPred[i] - hTarget[i])
		sumSq += diff * diff
	}
	meanLoss := float32(sumSq / float64(pred.Numel()))

	loss, err := NewFromFloat32(pred.ctx, []float32{meanLoss}, 1)
	if err != nil {
		return nil, err
	}

	if pred.RequiresGrad {
		loss.RequiresGrad = true
		loss.GradFn = &mseLossNode{out: loss, pred: pred, target: target}
	}

	return loss, nil
}
