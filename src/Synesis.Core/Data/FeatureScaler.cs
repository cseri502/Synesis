namespace Synesis.Core.Data;

public static class FeatureScaler
{
    public static void StandardizeInPlace(float[] flatData, int sampleCount, int featureCount)
    {
        for (int col = 0; col < featureCount; col++)
        {
            float mean = 0f;
            for (int row = 0; row < sampleCount; row++)
                mean += flatData[row * featureCount + col];
            
            mean /= sampleCount;
            
            float variance = 0f;
            for (int row = 0; row < sampleCount; row++)
            {
                float diff = flatData[row * featureCount + col] - mean;
                variance += diff * diff;
            }
            
            float std = MathF.Sqrt(variance / sampleCount);
            if (std < 1e-8f)
                std = 1e-8f;

            for (int row = 0; row < sampleCount; row++)
                flatData[row * featureCount + col] = (flatData[row * featureCount + col] - mean) / std;
        }
    }
}