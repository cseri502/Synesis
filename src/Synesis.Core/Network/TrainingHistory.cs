namespace Synesis.Core.Network;

public sealed class TrainingHistory
{
    private readonly List<(int Epoch, float Loss)> _records = [];
    public IReadOnlyList<(int Epoch, float Loss)> Records => _records;
    public void RecordEpoch(int epoch, float loss) => _records.Add((epoch, loss));
}