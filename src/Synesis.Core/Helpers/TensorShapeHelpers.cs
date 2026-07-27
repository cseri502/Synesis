namespace Synesis.Core.Helpers;

public class TensorShapeHelpers
{
    public static int ElementCount(ReadOnlySpan<nint> lengths)
    {
        int count = 1;
        
        foreach (var length in lengths)
            count *= (int)length;
        
        return count;
    }
}