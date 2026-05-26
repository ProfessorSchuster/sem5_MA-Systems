using UnityEngine;
using System.Collections;
using System.Collections.Generic;

[RequireComponent(typeof(MeshFilter))]

public class MatrixOperations : MonoBehaviour
{
    public Vector3 position = Vector3.zero;
    public Vector3 scale = Vector3.one;
    public Vector3 rotation = Vector3.zero;

    // Toggle
    public bool animate = true;
    public float speed = 1f;
    public float amplitude = 1f;

    // Rotate triangle
    public float triangleRadius = 1.5f;
    public float triangleSpinDegPerSec = 180f;
    public float triangleOrbitDegPerSec = 90f;
    public float triangleSize = 0.25f;

    Mesh tMesh;
    float tAngle;
    Transform tTransform;
    Material tMaterial;

    void Start()
    {
        GameObject triangle = new GameObject("Triangle");
        tTransform = triangle.transform;
        tTransform.position = Vector3.zero;
        tTransform.rotation = Quaternion.identity;

        var mfT = triangle.AddComponent<MeshFilter>();
        var mrT = triangle.AddComponent<MeshRenderer>();

        tMesh = CreateTriangleMesh(triangleSize);
        mfT.sharedMesh = tMesh;
        Shader sh = Shader.Find("Universal Render Pipeline/Lit");
        if (sh == null) sh = Shader.Find("Standard");
        tMaterial = new Material(sh);
        mrT.sharedMaterial = tMaterial;
    }

    void Update()
    {
        if (animate)
        {
            float t = Time.time * speed;
            position.x = Mathf.Cos(t) * amplitude;
            position.z = Mathf.Sin(t) * amplitude;
            rotation.y = t * 90f; // degrees/sec
        }

        Matrix4x4 translationMatrix = TranslateM(position.x, position.y, position.z);
        Matrix4x4 scaleMatrix = ScaleM(scale.x, scale.y, scale.z);
        Matrix4x4 rotationXMatrix = RotateX(rotation.x);
        Matrix4x4 rotationYMatrix = RotateY(rotation.y);
        Matrix4x4 rotationZMatrix = RotateZ(rotation.z);

        // Combined transformation matrix: T * Rz * Ry * Rx * S
        Matrix4x4 transformationMatrix = translationMatrix * rotationZMatrix * rotationYMatrix * rotationXMatrix * scaleMatrix;
        ApplyMatrixToTransform(transform, transformationMatrix);

        tAngle += triangleOrbitDegPerSec * Mathf.Deg2Rad * Time.deltaTime;
        float tx = Mathf.Cos(tAngle) * triangleRadius;
        float tz = Mathf.Sin(tAngle) * triangleRadius;
        tTransform.position = new Vector3(tx, 0f, tz);

        Vector3 forward = new Vector3(-Mathf.Sin(tAngle),0f, Mathf.Cos(tAngle)).normalized;
        if (forward.sqrMagnitude > 1e-6f) tTransform.rotation = Quaternion.LookRotation(forward, Vector3.up);
        tTransform.Rotate(Vector3.up, triangleSpinDegPerSec * Time.deltaTime, Space.Self);
    }

    Mesh CreateTriangleMesh(float size)
    {
        // Equilateral triangle centered near origin, flat on the floor (y=0)
        float s = Mathf.Max(1e-4f, size);
        // base coordinates in XZ (so it lies on floor); y=0
        Vector3 v0 = new Vector3(-0.5f * s, 0f, -Mathf.Sqrt(3f) / 6f * s);
        Vector3 v1 = new Vector3(0.5f * s, 0f, -Mathf.Sqrt(3f) / 6f * s);
        Vector3 v2 = new Vector3(0f, 0f, Mathf.Sqrt(3f) / 3f * s);

        Mesh m = new Mesh();
        m.name = "FloorTriangle";
        m.SetVertices(new List<Vector3> { v0, v1, v2 });
        m.SetTriangles(new int[] { 0, 1, 2 }, 0);
        m.RecalculateNormals();   // gives a (0,1,0) normal
        m.RecalculateBounds();
        return m;
    }

    // Translation
    public static Matrix4x4 TranslateM(float tx, float ty, float tz)
    {
        Matrix4x4 tm = Matrix4x4.identity;
        tm[0, 3] = tx; tm[1, 3] = ty; tm[2, 3] = tz;
        return tm;
    }
    // Scale
    public static Matrix4x4 ScaleM(float sx, float sy, float sz)
    {
        Matrix4x4 sm = Matrix4x4.identity;
        sm[0, 0] = sx;
        sm[1, 1] = sy;
        sm[2, 2] = sz;
        return sm;
    }

    public static Matrix4x4 RotateX(float angleDeg)
    {
        float r = angleDeg * Mathf.Deg2Rad;
        Matrix4x4 rx = Matrix4x4.identity;
        rx[1, 1] = Mathf.Cos(r);
        rx[1, 2] = -Mathf.Sin(r);
        rx[2, 1] = Mathf.Sin(r);
        rx[2, 2] = Mathf.Cos(r);
        return rx;
    }

    public static Matrix4x4 RotateY(float angleDeg)
    {
        float r = angleDeg * Mathf.Deg2Rad;
        Matrix4x4 ry = Matrix4x4.identity;
        ry[0, 0] = Mathf.Cos(r);
        ry[0, 2] = Mathf.Sin(r);
        ry[2, 0] = -Mathf.Sin(r);
        ry[2, 2] = Mathf.Cos(r);
        return ry;
    }

    public static Matrix4x4 RotateZ(float angleDeg)
    {
        float r = angleDeg * Mathf.Deg2Rad;
        Matrix4x4 rz = Matrix4x4.identity;
        rz[0, 0] = Mathf.Cos(r);
        rz[0, 1] = -Mathf.Sin(r);
        rz[1, 0] = Mathf.Sin(r);
        rz[1, 1] = Mathf.Cos(r);
        return rz;
    }

    // Apply transformation matrix to a Transform (extract pos/rot/scale)
    private static void ApplyMatrixToTransform(Transform tr, Matrix4x4 m)
    {
        // position = last column
        Vector3 pos = new Vector3(m[0, 3], m[1, 3], m[2, 3]);

        // columns 0..2 are basis vectors (with scale)
        Vector3 c0 = new Vector3(m[0, 0], m[1, 0], m[2, 0]);
        Vector3 c1 = new Vector3(m[0, 1], m[1, 1], m[2, 1]);
        Vector3 c2 = new Vector3(m[0, 2], m[1, 2], m[2, 2]);

        // scale = magnitudes
        float sx = c0.magnitude;
        float sy = c1.magnitude;
        float sz = c2.magnitude;

        // normalize to get pure rotation
        if (sx > 1e-6f) c0 /= sx;
        if (sy > 1e-6f) c1 /= sy;
        if (sz > 1e-6f) c2 /= sz;

        Quaternion rot = Quaternion.LookRotation(c2, c1);

        tr.position = pos;
        tr.rotation = rot;
        tr.localScale = new Vector3(sx, sy, sz);
    }
}
