using System.Numerics.Tensors;

namespace Synesis.Core.Activations;

public interface IActivationFunction
{
    Tensor<float> Forward(Tensor<float> input);
    Tensor<float> Derivative(Tensor<float> input);
}