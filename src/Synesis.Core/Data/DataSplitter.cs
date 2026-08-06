using System.Numerics.Tensors;

namespace Synesis.Core.Data;

public class DataSplitter
{
    public static (Tensor<float> TrainInputs, Tensor<float> TrainTargets,
        Tensor<float> TestInputs, Tensor<float> TestTargets)
        Split(Tensor<float> inputs, Tensor<float> targets, float testFraction = 0.2f, Random? random = null)
    {
        random ??= Random.Shared;

        int sampleCount = (int)inputs.Lengths[0];
        int inputFeatures = (int)inputs.Lengths[1];
        int targetFeatures = (int)targets.Lengths[1];

        var inputFlat = new float[sampleCount * inputFeatures];
        inputs.FlattenTo(inputFlat);
        var targetFlat = new float[sampleCount * targetFeatures];
        targets.FlattenTo(targetFlat);

        var indices = Enumerable.Range(0, sampleCount).ToArray();
        Shuffle(indices, random);

        int testCount = (int)(sampleCount * testFraction);
        int trainCount = sampleCount - testCount;

        return (
            Tensor.Create<float>(Gather(inputFlat, indices, 0, trainCount, inputFeatures), [(nint)trainCount, (nint)inputFeatures]),
            Tensor.Create<float>(Gather(targetFlat, indices, 0, trainCount, targetFeatures), [(nint)trainCount, (nint)targetFeatures]),
            Tensor.Create<float>(Gather(inputFlat, indices, trainCount, testCount, inputFeatures), [(nint)testCount, (nint)inputFeatures]),
            Tensor.Create<float>(Gather(targetFlat, indices, trainCount, testCount, targetFeatures), [(nint)testCount, (nint)targetFeatures])
        );
    }
    
    private static float[] Gather(float[] source, int[] indices, int start, int count, int featureSize)
    {
        var result = new float[count * featureSize];
        for (int i = 0; i < count; i++)
        {
            int sourceIndex = indices[start + i];
            Array.Copy(source, sourceIndex * featureSize, result, i * featureSize, featureSize);
        }
        return result;
    }

    private static void Shuffle(int[] array, Random random)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }
}