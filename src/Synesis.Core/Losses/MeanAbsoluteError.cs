using System.Numerics.Tensors;
using Synesis.Core.Helpers;

namespace Synesis.Core.Losses;

public sealed class MeanAbsoluteError : ILossFunction
{
    public float Compute(Tensor<float> predictions, Tensor<float> targets)
    {
        var diff = Tensor.Subtract<float>(predictions.AsReadOnlyTensorSpan(), targets.AsReadOnlyTensorSpan());
        var absDiff = Tensor.Abs<float>(diff.AsReadOnlyTensorSpan());
        
        float sum = Tensor.Sum<float>(absDiff.AsReadOnlyTensorSpan());
        return sum / TensorShapeHelpers.ElementCount(diff.Lengths);
    }

    public Tensor<float> Gradient(Tensor<float> predictions, Tensor<float> targets)
    {
        // dL/dpred = sign(pred - target) / N
        
        var diff = Tensor.Subtract<float>(predictions.AsReadOnlyTensorSpan(), targets.AsReadOnlyTensorSpan());
        int n = TensorShapeHelpers.ElementCount(diff.Lengths);
        
        var flat = new float[n];
        diff.FlattenTo(flat);
        
        for (int i = 0; i < flat.Length; i++)
            flat[i] = Math.Sign(flat[i]) / (float)n;
        
        return Tensor.Create(flat, diff.Lengths);
    }
}