using System.Numerics.Tensors;

namespace Synesis.Core.Losses;

public sealed class SoftmaxCrossEntropyLoss : ILossFunction
{
    private const float Epsilon = 1e-7f;

    public float Compute(Tensor<float> logits, Tensor<float> targets)
    {
        int batchSize = (int)logits.Lengths[0];
        int classCount = (int)logits.Lengths[1];

        var logitsFlat = new float[batchSize * classCount];
        var targetsFlat = new float[batchSize * classCount];
        logits.FlattenTo(logitsFlat);
        targets.FlattenTo(targetsFlat);

        float totalLoss = 0f;

        for (int b = 0; b < batchSize; b++)
        {
            int offset = b * classCount;
            
            float maxLogit = float.NegativeInfinity;
            for (int c = 0; c < classCount; c++)
                if (logitsFlat[offset + c] > maxLogit) maxLogit = logitsFlat[offset + c];

            float sumExp = 0f;
            for (int c = 0; c < classCount; c++)
                sumExp += MathF.Exp(logitsFlat[offset + c] - maxLogit);

            for (int c = 0; c < classCount; c++)
            {
                float prob = MathF.Exp(logitsFlat[offset + c] - maxLogit) / sumExp;
                float target = targetsFlat[offset + c];
                totalLoss -= target * MathF.Log(prob + Epsilon);
            }
        }

        return totalLoss / batchSize;
    }

    public Tensor<float> Gradient(Tensor<float> logits, Tensor<float> targets)
    {
        int batchSize = (int)logits.Lengths[0];
        int classCount = (int)logits.Lengths[1];

        var logitsFlat = new float[batchSize * classCount];
        var targetsFlat = new float[batchSize * classCount];
        logits.FlattenTo(logitsFlat);
        targets.FlattenTo(targetsFlat);

        var gradFlat = new float[batchSize * classCount];
        float invBatch = 1f / batchSize;

        for (int b = 0; b < batchSize; b++)
        {
            int offset = b * classCount;

            float maxLogit = float.NegativeInfinity;
            for (int c = 0; c < classCount; c++)
                if (logitsFlat[offset + c] > maxLogit) maxLogit = logitsFlat[offset + c];

            float sumExp = 0f;
            for (int c = 0; c < classCount; c++)
                sumExp += MathF.Exp(logitsFlat[offset + c] - maxLogit);

            for (int c = 0; c < classCount; c++)
            {
                float prob = MathF.Exp(logitsFlat[offset + c] - maxLogit) / sumExp;
                float target = targetsFlat[offset + c];
                
                // dL/dz = (p - y) / BatchSize
                gradFlat[offset + c] = (prob - target) * invBatch;
            }
        }

        return Tensor.Create(gradFlat, logits.Lengths);
    }
}