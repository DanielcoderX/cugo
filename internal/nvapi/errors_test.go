//go:build windows

package nvapi

import (
	"strings"
	"testing"
)

func TestResultToError(t *testing.T) {
	if err := ResultToError(CUDA_SUCCESS); err != nil {
		t.Fatalf("expected nil error for CUDA_SUCCESS, got: %v", err)
	}

	err := ResultToError(CUDA_ERROR_OUT_OF_MEMORY)
	if err == nil {
		t.Fatal("expected error for CUDA_ERROR_OUT_OF_MEMORY, got nil")
	}

	if !strings.Contains(err.Error(), "CUDA_ERROR_OUT_OF_MEMORY") {
		t.Fatalf("expected error string to contain 'CUDA_ERROR_OUT_OF_MEMORY', got: %q", err.Error())
	}

	unknownErr := ResultToError(CUresult(99999))
	if !strings.Contains(unknownErr.Error(), "CUDA_ERROR_UNKNOWN (99999)") {
		t.Fatalf("expected unknown error formatting, got: %q", unknownErr.Error())
	}
}
