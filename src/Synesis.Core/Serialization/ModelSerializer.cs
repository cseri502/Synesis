using System.Text.Json;
using Synesis.Core.Layers;
using Synesis.Core.Network;

namespace Synesis.Core.Serialization;

public static class ModelSerializer
{
    private sealed class SerializedModel
    {
        public List<Dictionary<string, float[]>> Layers { get; set; } = new();
    }
    
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    public static void Save(NeuralNetwork network, string filePath)
    {
        var layers = network.Layers
            .Select(layer => layer is IParameterizedLayer parameterized 
                ? parameterized.GetParameters().ToDictionary(p => p.Name, p => p.Values)
                : new Dictionary<string, float[]>()
            ).ToList();
        
        var model = new SerializedModel() { Layers = layers };
        File.WriteAllText(filePath, JsonSerializer.Serialize(model, JsonOptions));
    }

    public static void Load(NeuralNetwork network, string filePath)
    {
        var json = File.ReadAllText(filePath);
        var model = JsonSerializer.Deserialize<SerializedModel>(json, JsonOptions) 
                    ?? throw new InvalidDataException("Could not deserialize model.");
        
        if (model.Layers.Count != network.Layers.Count)
            throw new InvalidDataException("Layer count mismatch.");

        for (int i = 0; i < network.Layers.Count; i++)
        {
            if (network.Layers[i] is IParameterizedLayer parameterized)
                parameterized.SetParameters(model.Layers[i]);
        }
    }
}