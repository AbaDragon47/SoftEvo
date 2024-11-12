using UnityEngine;

public class ActuatorController : MonoBehaviour
{
    public MeshDeformer meshDeformer;
    
    public float inflationForce = 1f; // Inflation force strength (tune this)
    
    void Start()
    {
        // Get the MeshDeformer component from the child (starfish)
        meshDeformer = GetComponentInChildren<MeshDeformer>();
    }
    void Update()
    {
        // Simulate pneumatic actuation for each arm with key presses
        if (Input.GetKeyDown(KeyCode.Alpha1)) // Activate arm 0 (e.g., press '1' to inflate arm 1)
        {
            meshDeformer.ApplyPneumaticForce(0, inflationForce);
        }
        if (Input.GetKeyDown(KeyCode.Alpha2)) // Activate arm 1 (e.g., press '2' to inflate arm 2)
        {
            meshDeformer.ApplyPneumaticForce(1, inflationForce);
        }
        if (Input.GetKeyDown(KeyCode.Alpha3)) // Activate arm 2
        {
            meshDeformer.ApplyPneumaticForce(2, inflationForce);
        }
        if (Input.GetKeyDown(KeyCode.Alpha4)) // Activate arm 3
        {
            meshDeformer.ApplyPneumaticForce(3, inflationForce);
        }
        if (Input.GetKeyDown(KeyCode.Alpha5)) // Activate arm 4
        {
            meshDeformer.ApplyPneumaticForce(4, inflationForce);
        }
    }
}
