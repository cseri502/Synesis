using System.Numerics.Tensors;

namespace Synesis.Core.Activations;

public sealed class Softmax : IActivationFunction
{
    public Tensor<float> Forward(Tensor<float> input) => Tensor.SoftMax<float>(input.AsReadOnlyTensorSpan());

    public Tensor<float> Derivative(Tensor<float> input)
    {
        throw new NotSupportedException();
    }
}