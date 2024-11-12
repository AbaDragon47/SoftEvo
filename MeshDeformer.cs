using UnityEngine;
using System.Collections.Generic;


[RequireComponent(typeof(MeshFilter))]
public class MeshDeformer : MonoBehaviour {

	public float springForce = 20f;
	public float damping = 5f;

	Mesh deformingMesh;
	Vector3[] originalVertices, displacedVertices;
	Vector3[] vertexVelocities;
	Vector3 meshCenter;

	float uniformScale = 1f, armRadius;

	public Vector3[] armCenters;

	/*List<int> arm1Vertices = new List<int>();
    	List<int> arm2Vertices = new List<int>();
    	List<int> arm3Vertices = new List<int>();
    	List<int> arm4Vertices = new List<int>();
    	List<int> arm5Vertices = new List<int>();*/

	void Start () {

		
		deformingMesh = GetComponent<MeshFilter>().mesh = Instantiate(GetComponent<MeshFilter>().mesh);
    		originalVertices = deformingMesh.vertices; //shallow copy fn
		displacedVertices = new Vector3[originalVertices.Length];
		/*for (int i = 0; i < originalVertices.Length; i++) {
			displacedVertices[i] = originalVertices[i];
		}*/
		vertexVelocities = new Vector3[originalVertices.Length];
		CalculateMeshCenterAndRadius();
		armCenters = CalculateArmCenters(5);
		VisualizeArmCenters();
		

		

		//AssignArmVertices();
	}
	Vector3[] CalculateArmCenters(int armsCount) {
		Vector3[] armPositions = new Vector3[armsCount];

		
		float angleIncrement = 360f / armsCount;  

		
		for (int i = 0; i < armsCount; i++) {
			
			/*float angle = Mathf.Deg2Rad * i * angleIncrement;

			// The arm will be placed at the calculated armRadius distance from the center.
			float x = Mathf.Cos(angle) * armRadius; 
			float z = Mathf.Sin(angle) * armRadius;

			armPositions[i] = new Vector3(x, 0f, z);*/
			GameObject[] actuatorObjects = GameObject.FindGameObjectsWithTag("actuator");

			armCenters = new Vector3[actuatorObjects.Length];

			// Store the position of each actuator in armCenters
			armPositions[i] = actuatorObjects[i].transform.position;
			Debug.Log("Arm " + i + " position: " + armPositions[i]);
		

		}


		return armPositions;
	}

	void VisualizeArmCenters() {
		GameObject sphere;
		Renderer sphereRenderer;
		for (int i = 0; i < armCenters.Length; i++) {

			sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
			sphere.transform.position = armCenters[i];

			sphere.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);

			sphereRenderer = sphere.GetComponent<Renderer>();
			sphereRenderer.material.color = Color.red;


			sphere.name = "ArmCenter_" + i;
		}
		sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
		sphere.transform.position = meshCenter;


		sphere.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);


		sphereRenderer = sphere.GetComponent<Renderer>();
		sphereRenderer.material.color = Color.red;

		sphere.name = "meshcenter";
	}


	void Update () {

		 if (deformingMesh == null) {
			Debug.LogError("Deforming mesh is null, cannot update vertices.");
			return;
		}

		uniformScale = transform.localScale.x;
		for (int i = 0; i < displacedVertices.Length; i++) {
			UpdateVertex(i);
		}
		deformingMesh.vertices = displacedVertices;
		deformingMesh.RecalculateNormals();
	}

	void UpdateVertex (int i) {
		Vector3 velocity = vertexVelocities[i];
		Vector3 displacement = displacedVertices[i] - originalVertices[i];
		displacement *= uniformScale;
		velocity -= displacement * springForce * Time.deltaTime;
		velocity *= 1f - damping * Time.deltaTime;
		vertexVelocities[i] = velocity;
		displacedVertices[i] += velocity * (Time.deltaTime / uniformScale);
	}

	public void AddDeformingForce (Vector3 point, float force, int armIndex) {
		point = transform.InverseTransformPoint(point);
		/*List<int> armVertices = GetArmVertices(armIndex);

		foreach (int i in armVertices) {
            AddForceToVertex(i, point, force);
        	}*/
		for (int i = 0; i < displacedVertices.Length; i++) {
			AddForceToVertex(i, point, force);
		}
	}

	void AddForceToVertex (int i, Vector3 point, float force) {
		Vector3 pointToVertex = displacedVertices[i] - point;
		pointToVertex *= uniformScale;
		float attenuatedForce = force / (1f + pointToVertex.sqrMagnitude);
		float velocity = attenuatedForce * Time.deltaTime;
		vertexVelocities[i] += pointToVertex.normalized * velocity;
	}

	public void ApplyPneumaticForce(int armIndex, float inflationForce)
    {
		if (armIndex < 0 || armIndex >= armCenters.Length)
		{
			Debug.LogError("Invalid arm index: " + armIndex);
			return;
		}
		Vector3 inflationCenter = armCenters[armIndex]; 
		for (int i = 0; i < displacedVertices.Length; i++)
		{
			if (IsVertexInArm(i, armIndex))
			{
				Vector3 direction = displacedVertices[i] - inflationCenter;
				float distance = direction.magnitude;
				float inflationAmount = Mathf.Exp(-distance * inflationForce); 

				displacedVertices[i] += direction.normalized * inflationAmount * Time.deltaTime;
			}
		}
		deformingMesh.vertices = displacedVertices;
		deformingMesh.RecalculateNormals();
    }
	public void ApplyBendingForce(int armIndex, float bendingForce) {
		Vector3 armDirection = armCenters[armIndex] - meshCenter;
		for (int i = 0; i < displacedVertices.Length; i++) {
			if (IsVertexInArm(i, armIndex)) {
				// Apply bending force along the arm
				Vector3 vertexDirection = displacedVertices[i] - meshCenter;
				Vector3 bendingDirection = Vector3.Cross(armDirection, vertexDirection);
				displacedVertices[i] += bendingDirection.normalized * bendingForce * Time.deltaTime;
			}
		}
		deformingMesh.vertices = displacedVertices;
		deformingMesh.RecalculateNormals();
	}


    bool IsVertexInArm(int vertexIndex, int armIndex)
    {
        // Implement logic to check if a vertex belongs to a specific arm
        // You can use a spatial comparison or bounding box based on your arm setup
        // For simplicity, let's assume all vertices related to an arm are near its center
        return true; // This is a placeholder
    }

    


	void CalculateMeshCenterAndRadius() {
        Vector3 sum = Vector3.zero;
        float maxDistance = 0f;

        foreach (Vector3 vertex in originalVertices) {
            sum += vertex;
        }
	   Debug.Log("Sum "+sum);
        meshCenter = sum / originalVertices.Length;

        foreach (Vector3 vertex in originalVertices) {
            float distance = Vector3.Distance(vertex, meshCenter);
            maxDistance = Mathf.Max(maxDistance, distance);
        }

        armRadius = maxDistance; 
        Debug.Log("Mesh Center: " + meshCenter);
        Debug.Log("Arm Radius: " + armRadius);
    }

/*	void AssignArmVertices() {
        float angleIncrement = Mathf.PI * 2f / 5f; 

        for (int i = 0; i < originalVertices.Length; i++) {
            Vector3 vertex = originalVertices[i];
		  Vector3 direction = vertex - meshCenter;
            float angle = Mathf.Atan2(direction.z, direction.x) * Mathf.Rad2Deg;
            angle = (angle + Mathf.PI * 2f) % (Mathf.PI * 2f); // Normalize the angle to be between 0 and 2PI


            for (int armIndex = 0; armIndex < 5; armIndex++) {
                float lowerBound = armIndex * angleIncrement;
                float upperBound = (armIndex + 1) * angleIncrement;

                if (angle >= lowerBound && angle < upperBound) {
                    switch (armIndex) {
                        case 0: arm1Vertices.Add(i); break;
                        case 1: arm2Vertices.Add(i); break;
                        case 2: arm3Vertices.Add(i); break;
                        case 3: arm4Vertices.Add(i); break;
                        case 4: arm5Vertices.Add(i); break;
                    }
                    break;
                }
            }
        }
    }
    List<int> GetArmVertices(int armIndex) {
        switch (armIndex) {
            case 0: return arm1Vertices;
            case 1: return arm2Vertices;
            case 2: return arm3Vertices;
            case 3: return arm4Vertices;
            case 4: return arm5Vertices;

            default: return arm1Vertices;
        }
    }

*/


}