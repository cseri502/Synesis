using System.Numerics.Tensors;
using Synesis.Core.Helpers;

namespace Synesis.Core.Losses;

public class MeanSquaredError : ILossFunction
{
    public float Compute(Tensor<float> predictions, Tensor<float> targets)
    {
        var diff = Tensor.Subtract<float>(predictions.AsReadOnlyTensorSpan(), targets.AsReadOnlyTensorSpan());
        var squared = Tensor.Multiply<float>(diff.AsReadOnlyTensorSpan(), diff.AsReadOnlyTensorSpan());
        
        float sum = Tensor.Sum<float>(squared.AsReadOnlyTensorSpan());
        return sum / TensorShapeHelpers.ElementCount(diff.Lengths);
    }

    public Tensor<float> Gradient(Tensor<float> predictions, Tensor<float> targets)
    {
        // dL/dpred = 2 * (pred - target) / N
        
        var diff = Tensor.Subtract<float>(predictions.AsReadOnlyTensorSpan(), targets.AsReadOnlyTensorSpan());
        int n = TensorShapeHelpers.ElementCount(diff.Lengths);
        
        return Tensor.Multiply(diff.AsReadOnlyTensorSpan(), 2f / n);
    }
}