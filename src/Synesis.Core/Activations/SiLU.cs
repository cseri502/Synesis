using System.Numerics.Tensors;

namespace Synesis.Core.Activations;

public sealed class SiLU : IActivationFunction
{
    public Tensor<float> Forward(Tensor<float> input)
    {
        // f(x) = x * sigmoid(x)
        var sig = Tensor.Sigmoid<float>(input.AsReadOnlyTensorSpan());
        return Tensor.Multiply<float>(input.AsReadOnlyTensorSpan(), sig.AsReadOnlyTensorSpan());
    }

    public Tensor<float> Derivative(Tensor<float> input)
    {
        // f'(x) = sig(x) + f(x) * (1 - sig(x))
        var sig = Tensor.Sigmoid<float>(input.AsReadOnlyTensorSpan());
        var silu = Forward(input);
        
        // 1 - sig(x)
        var oneMinusSig = Tensor.Subtract<float>(1f, sig.AsReadOnlyTensorSpan());
        
        // silu * (1 - sig(x))
        var term2 = Tensor.Multiply<float>(silu.AsReadOnlyTensorSpan(), oneMinusSig.AsReadOnlyTensorSpan());
        
        // sig(x) + term2
        return Tensor.Add<float>(sig.AsReadOnlyTensorSpan(), term2.AsReadOnlyTensorSpan());
    }
}