using System.Globalization;
using System.Numerics.Tensors;

namespace WineDemo;

public static class WineDatasetLoader
{
    private const int FeatureCount = 13;
    private const int ClassCount = 3;

    public static (Tensor<float> Inputs, Tensor<float> Targets) Load(string csvPath)
    {
        var lines = File.ReadAllLines(csvPath);
        int sampleCount = lines.Length;

        var inputData = new float[sampleCount * FeatureCount];
        var targetData = new float[sampleCount * ClassCount];

        for (int row = 0; row < sampleCount; row++)
        {
            var parts = lines[row].Split(',');

            int classLabel = int.Parse(parts[0], CultureInfo.InvariantCulture) - 1;

            for (int col = 0; col < FeatureCount; col++)
                inputData[row * FeatureCount + col] = float.Parse(parts[col + 1], CultureInfo.InvariantCulture);

            targetData[row * ClassCount + classLabel] = 1f;
        }

        var inputs = Tensor.Create<float>(inputData, [(nint)sampleCount, (nint)FeatureCount]);
        var targets = Tensor.Create<float>(targetData, [(nint)sampleCount, (nint)ClassCount]);

        return (inputs, targets);
    }
}