using UnityEngine;

public class cube_mesh_script : MonoBehaviour
{
    Mesh mesh;
    Vector3[] vertices;
    int[] triangles;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        mesh = new Mesh();
        GetComponent<MeshFilter>().mesh = mesh;

        CreateShape();
        UpdateMesh();
    }

    // Update is called once per frame
    void Update()
    {
        mesh.Clear();
        mesh.vertices = vertices;
        mesh.triangles = triangles;
    }

    void CreateShape()
    {
        vertices = new Vector3[] {
        new Vector3(0, 0, 0), // lower left back
        new Vector3(1, 0, 0), // lower right back
        new Vector3(1, 1, 0), // upper right back
        new Vector3(0, 1, 0), // upper left back

        new Vector3(0, 0, 1), // lower left front
        new Vector3(1, 0, 1), // lower right front
        new Vector3(1, 1, 1), // upper right front
        new Vector3(0, 1, 1)  // upper left front
        };

        triangles = new int[]
        {
        3, 2, 1, // front face
        3, 1, 0, // front face
        4, 1, 5, // bottom face
        4, 0, 1, // bottom face
        7, 2, 3, // top face
        7, 6, 2, // top face
        2, 6, 5, // right face
        2, 5, 1, // right face
        3, 0, 4, // left face
        3, 4, 7, // left face
        6, 7, 4, // back face
        6, 4, 5  // back face
        };
    }

    void UpdateMesh()
    {
        mesh.Clear();

        mesh.vertices = vertices;
        mesh.triangles = triangles;
    }

}
