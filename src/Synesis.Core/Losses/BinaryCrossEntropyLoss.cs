using System.Numerics.Tensors;

namespace Synesis.Core.Losses;

public sealed class BinaryCrossEntropyLoss : ILossFunction
{
    private const float Epsilon = 1e-7f;

    public float Compute(Tensor<float> predictions, Tensor<float> targets)
    {
        int totalElements = (int)predictions.Lengths.ToArray().Aggregate(1, (a, b) => a * (int)b);

        var pFlat = new float[totalElements];
        var tFlat = new float[totalElements];
        predictions.FlattenTo(pFlat);
        targets.FlattenTo(tFlat);

        float sum = 0f;
        for (int i = 0; i < totalElements; i++)
        {
            float p = Math.Clamp(pFlat[i], Epsilon, 1f - Epsilon);
            float y = tFlat[i];
            sum -= y * MathF.Log(p) + (1f - y) * MathF.Log(1f - p);
        }

        return sum / totalElements;
    }

    public Tensor<float> Gradient(Tensor<float> predictions, Tensor<float> targets)
    {
        int totalElements = (int)predictions.Lengths.ToArray().Aggregate(1, (a, b) => a * (int)b);

        var pFlat = new float[totalElements];
        var tFlat = new float[totalElements];
        predictions.FlattenTo(pFlat);
        targets.FlattenTo(tFlat);

        var gradFlat = new float[totalElements];
        float invN = 1f / totalElements;

        for (int i = 0; i < totalElements; i++)
        {
            float p = Math.Clamp(pFlat[i], Epsilon, 1f - Epsilon);
            float y = tFlat[i];

            // dL/dp = (p - y) / (p * (1 - p))
            gradFlat[i] = ((p - y) / (p * (1f - p))) * invN;
        }

        return Tensor.Create(gradFlat, predictions.Lengths);
    }
}