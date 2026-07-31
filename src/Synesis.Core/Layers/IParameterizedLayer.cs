namespace Synesis.Core.Layers;

public interface IParameterizedLayer
{
    IEnumerable<(string Name, float[] Values)> GetParameters();
    void SetParameters(IReadOnlyDictionary<string, float[]> parameters);
}