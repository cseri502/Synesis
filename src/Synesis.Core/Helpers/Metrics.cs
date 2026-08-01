using System.Numerics.Tensors;

namespace Synesis.Core.Helpers;

public static class Metrics
{
    public static float Accuracy(Tensor<float> predictions, Tensor<float> targets)
    {
        int batchSize = (int)predictions.Lengths[0];
        int classCount = (int)predictions.Lengths[1];
        
        var predFlat = new float[batchSize * classCount];
        var targetFlat = new float[batchSize * classCount];
        predictions.FlattenTo(predFlat);
        targets.FlattenTo(targetFlat);

        int correct = 0;
        
        for (int b = 0; b < batchSize; b++)
        {
            int predictedClass = ArgMax(predFlat, b * classCount, classCount);
            int actualClass = ArgMax(targetFlat, b * classCount, classCount);

            if (predictedClass == actualClass)
                correct++;
        }

        return (float)correct / batchSize;
    }
    
    private static int ArgMax(float[] data, int offset, int count)
    {
        int bestIndex = 0;
        float bestValue = data[offset];

        for (int i = 1; i < count; i++)
        {
            if (data[offset + i] > bestValue)
            {
                bestValue = data[offset + i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }
}