package tensor

import (
	"errors"
	"fmt"
)

// Node represents an operation in the reverse-mode automatic differentiation graph.
type Node interface {
	// Backward propagates output gradients to its inputs.
	Backward() error
	// Inputs returns the input tensors to this operation that require gradients.
	Inputs() []*Tensor
}

// Backward executes reverse-mode automatic differentiation starting from this tensor.
// It computes gradients for all leaf tensors in the computation graph that have RequiresGrad = true.
func (t *Tensor) Backward() error {
	if t == nil || t.storage == nil {
		return errors.New("cugo/autograd: cannot compute backward on nil or freed tensor")
	}

	// 1. If this tensor has no incoming grad, initialize seed gradient to 1.0
	if t.Grad == nil {
		ones := make([]float32, t.Numel())
		for i := range ones {
			ones[i] = 1.0
		}
		seedGrad, err := NewFromFloat32(t.ctx, ones, t.shape...)
		if err != nil {
			return fmt.Errorf("cugo/autograd: failed to create seed gradient: %w", err)
		}
		t.Grad = seedGrad
	}

	if t.GradFn == nil {
		// Leaf tensor with no creators
		return nil
	}

	// 2. Build topological ordering of computation nodes via DFS
	var order []Node
	visited := make(map[Node]bool)

	var buildTopo func(n Node)
	buildTopo = func(n Node) {
		if n == nil || visited[n] {
			return
		}
		visited[n] = true
		for _, inp := range n.Inputs() {
			if inp != nil && inp.GradFn != nil {
				buildTopo(inp.GradFn)
			}
		}
		order = append(order, n)
	}

	buildTopo(t.GradFn)

	// 3. Execute backward passes in reverse topological order
	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		if err := node.Backward(); err != nil {
			return fmt.Errorf("cugo/autograd: node backward failed: %w", err)
		}
	}

	return nil
}
