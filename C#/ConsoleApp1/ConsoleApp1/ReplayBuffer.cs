namespace ConsoleApp1;

public class ReplayBuffer(int capacity, int steps = 1, float gamma = 0.99f)
{
    public int Capacity { get; private set; } = capacity;
    public int Steps { get; private set; } = steps;
    public float Gamma { get; private set; } = gamma;
    public LinkedList<Step> Buffer { get; private set; } = []; // TODO: replace dynamic with the state tuple
    // TODO: make sure the max length is capacity
    public LinkedList<Step> NStepBuffer { get; private set; } = [];
    // TODO: make sure the max length is steps

    void Add(Step transition)
    {
        if (Steps == 1)
        {
            Buffer.Append(transition);
            if (Buffer.Count > Capacity)
                Buffer.RemoveFirst();
            return;
        }

        NStepBuffer.Append(transition);
        var (_, _, _, finalState, finalTermination) = transition;
        float nStepReward = 0f;

        foreach (var (_, _, reward, _, _) in NStepBuffer.Reverse())
            nStepReward = (nStepReward * Gamma) + reward;

        var (state, action, _, _, _) = NStepBuffer.First();
        
        if (NStepBuffer.Count == Steps)
            Buffer.Append(new(state, action, nStepReward, finalState, finalTermination));
        
        if (finalTermination)
            NStepBuffer.Clear();
    }

    Tuple<List<dynamic>, List<dynamic>, List<float>, List<dynamic>, List<bool>> Sample(int batchSize)
    {
        Random random = new();
        var sampledItems = Buffer.OrderBy(x => random.Next()).Take(batchSize).ToList();
        var states = sampledItems.Select(x => x.State).ToList();
        var actions = sampledItems.Select(x => x.Action).ToList();
        var rewards = sampledItems.Select(x => x.Reward).ToList();
        var nextStates = sampledItems.Select(x => x.NextState).ToList();
        var terminations = sampledItems.Select(x => x.Terminated).ToList();
        return Tuple.Create(states, actions, rewards, nextStates, terminations);
    }
    
    public int Count => Buffer.Count;
}