using System.Numerics.Tensors;

namespace Synesis.Core.Losses;

public sealed class SoftmaxCrossEntropyLoss : ILossFunction
{
    private const float Epsilon = 1e-7f;

    public float Compute(Tensor<float> logits, Tensor<float> targets)
    {
        var probabilities = Tensor.SoftMax<float>(logits.AsReadOnlyTensorSpan());
        
        return CrossEntropy(probabilities, targets);
    }

    public Tensor<float> Gradient(Tensor<float> logits, Tensor<float> targets)
    {
        var probabilities = Tensor.SoftMax<float>(logits.AsReadOnlyTensorSpan());
        int batchSize = (int)logits.Lengths[0];
        
        var diff = Tensor.Subtract<float>(probabilities.AsReadOnlyTensorSpan(), targets.AsReadOnlyTensorSpan());
        return Tensor.Multiply<float>(diff.AsReadOnlyTensorSpan(), 1f / batchSize);
    }

    private static float CrossEntropy(Tensor<float> probabilities, Tensor<float> targets)
    {
        int batchSize = (int)probabilities.Lengths[0];
        int classCount = (int)probabilities.Lengths[1];

        var probFlat = new float[batchSize * classCount];
        var targetFlat = new float[batchSize * classCount];
        probabilities.FlattenTo(probFlat);
        targets.FlattenTo(targetFlat);

        float sum = 0f;
        for (int i = 0; i < probFlat.Length; i++)
            sum -= targetFlat[i] * MathF.Log(probFlat[i] + Epsilon);

        return sum / batchSize;
    }
}