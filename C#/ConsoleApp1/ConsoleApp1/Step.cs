namespace ConsoleApp1;

public struct Step(dynamic state, dynamic action, float reward, dynamic nextState, bool terminated)
{
    public dynamic State = state;
    public dynamic Action = action;
    public float Reward = reward;
    public dynamic NextState = nextState;
    public bool Terminated = terminated;

    public void Deconstruct(out dynamic state, out dynamic action, out float reward, out dynamic nextState, out bool terminated)
    {
        state = State;
        action = Action;
        reward = Reward;
        nextState = NextState;
        terminated = Terminated;
    }

    public int Count = 5;
}