using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class WindowMarker : MonoBehaviour { }

[ExecuteAlways]
public class SphereWindowGenerator : MonoBehaviour
{
    [Header("References")]
    public GameObject windowPrefab;
    public Transform sphereCenter;

    [Header("Sphere")]
    [Min(0.001f)]
    public float sphereRadius = 10f;

    [Header("Layout")]
    [Min(1)] public int ringCount = 6;
    [Min(1)] public int windowsPerRing = 24;
    [Range(-90f, 90f)] public float minLatitudeDeg = -15f;
    [Range(-90f, 90f)] public float maxLatitudeDeg = +15f;
    public float ringPhaseOffset = 0.5f;

    [Header("Window Sizing / Placement")]
    [Min(0f)] public float radialOffset = 0.02f;
    public Vector2 windowSize = new Vector2(0.9f, 0.6f);
    public Vector3 localEulerOffset = Vector3.zero;

    [Header("Parenting & Naming")]
    public string windowsRootName = "Windows";
    public string namePattern = "Window_Ring{r}_{i}";

    [Header("Editor")]
    public bool autoRegenerate = true;
    public bool drawGizmos = false;

    [Header("Orientation (Prefab path only)")]
    public bool usingPlanePrefab = false;
    public Vector3 localEulerOffsetForPlane = new Vector3(-90f, 0f, 0f);
    public bool flipNormals = false;

    [Header("Curved Windows")]
    [Tooltip("Erzeugt pro Fenster ein gekrümmtes Mesh, das exakt auf der Kugel liegt.")]
    public bool buildCurvedWindows = true;

    [Tooltip("Segmente horizontal (links↔rechts) für die Krümmung.")]
    [Min(1)] public int segmentsU = 10;

    [Tooltip("Segmente vertikal (unten↔oben) für die Krümmung.")]
    [Min(1)] public int segmentsV = 6;

    private Transform windowsRoot;

#if UNITY_EDITOR
    [ContextMenu("Generate Now")]
#endif
    public void Generate()
    {
        if (windowPrefab == null || sphereCenter == null)
        {
            Debug.LogWarning("[SphereWindowGenerator] Bitte Window Prefab und Sphere Center zuweisen.");
            return;
        }

        WarnIfNegativeScale(transform);
        WarnIfNegativeScale(sphereCenter);

        EnsureWindowsRoot();
        ClearWindowsImmediate();

        int rings = Mathf.Max(1, ringCount);
        int perRing = Mathf.Max(1, windowsPerRing);

        float minLat = Mathf.Deg2Rad * minLatitudeDeg;
        float maxLat = Mathf.Deg2Rad * maxLatitudeDeg;

        // Material(s) vom Prefab (für den curved mode übernehmen)
        Material[] prefabMats = GetPrefabMaterials(windowPrefab);

        for (int r = 0; r < rings; r++)
        {
            float t = (rings == 1) ? 0.5f : (float)r / (rings - 1); // 0..1
            float phiCenter = Mathf.Lerp(minLat, maxLat, t);        // Breitengrad in RAD
            float phase = ringPhaseOffset * 2f * Mathf.PI / perRing;

            for (int i = 0; i < perRing; i++)
            {
                float lambdaCenter = (i * 2f * Mathf.PI / perRing) + phase * r; // Längengrad in RAD

                // Basis: Mittelpunkt auf der Kugel
                Vector3 centerLocal = SphericalToCartesian(sphereRadius + radialOffset, phiCenter, lambdaCenter);
                Vector3 centerWorld = sphereCenter.TransformPoint(centerLocal);

                // Oberflächennormale (nach außen) + Tangenten
                Vector3 nWorld = (centerWorld - sphereCenter.position).normalized;
                // Nord (Breitengrad-Tangente)
                Vector3 tPhiLocal = new Vector3(
                    -Mathf.Sin(phiCenter) * Mathf.Cos(lambdaCenter),
                     Mathf.Cos(phiCenter),
                    -Mathf.Sin(phiCenter) * Mathf.Sin(lambdaCenter)
                ).normalized;
                Vector3 northWorld = sphereCenter.TransformDirection(tPhiLocal).normalized;
                // Ost (Längengrad-Tangente)
                Vector3 eastWorld = Vector3.Cross(nWorld, northWorld).normalized;

                if (buildCurvedWindows)
                {
                    // Erzeuge gekrümmten Fenster-Patch
                    GameObject w = CreateCurvedWindowPatch(
                        centerWorld, nWorld, eastWorld, northWorld,
                        phiCenter, lambdaCenter, prefabMats
                    );

                    // Name & Marker
                    if (w.GetComponent<WindowMarker>() == null) w.AddComponent<WindowMarker>();
                    w.name = namePattern.Replace("{r}", (r + 1).ToString()).Replace("{i}", (i + 1).ToString());
                    w.transform.SetParent(windowsRoot, true);
                }
                else
                {
                    // Fallback: vorhandenes Prefab instanzieren + ausrichten
                    Quaternion rot = Quaternion.LookRotation(nWorld, northWorld);
                    Quaternion extraRot = Quaternion.identity;
                    if (usingPlanePrefab) extraRot *= Quaternion.Euler(localEulerOffsetForPlane);
                    if (flipNormals) extraRot *= Quaternion.Euler(0f, 180f, 0f);
                    extraRot *= Quaternion.Euler(localEulerOffset);

                    GameObject w = InstantiateSafe(windowPrefab, centerWorld, rot * extraRot, windowsRoot);
                    if (w.GetComponent<WindowMarker>() == null) w.AddComponent<WindowMarker>();
                    w.transform.localScale = new Vector3(windowSize.x, windowSize.y, 1f);
                    w.name = namePattern.Replace("{r}", (r + 1).ToString()).Replace("{i}", (i + 1).ToString());
                }
            }
        }
    }

    // --- Curved Mesh creation ---

    private GameObject CreateCurvedWindowPatch(
        Vector3 centerWorld,
        Vector3 nWorld,
        Vector3 eastWorld,
        Vector3 northWorld,
        float phiCenter,
        float lambdaCenter,
        Material[] materials
    )
    {
        // Neues GO mit Identity, damit wir world→local sauber konvertieren können
        GameObject go = new GameObject("CurvedWindow");
        go.transform.position = centerWorld;
        go.transform.rotation = Quaternion.LookRotation(nWorld, northWorld);
        go.transform.localScale = Vector3.one;

        var mf = go.AddComponent<MeshFilter>();
        var mr = go.AddComponent<MeshRenderer>();
        if (materials != null && materials.Length > 0) mr.sharedMaterials = materials;

        Mesh mesh = new Mesh();
        mf.sharedMesh = mesh;

        int nx = Mathf.Max(1, segmentsU);
        int ny = Mathf.Max(1, segmentsV);
        int vx = nx + 1;
        int vy = ny + 1;

        Vector3[] verts = new Vector3[vx * vy];
        Vector2[] uvs = new Vector2[verts.Length];
        int[] tris = new int[nx * ny * 6];

        // Physische Fenster-Halbmaße
        float halfW = windowSize.x * 0.5f;
        float halfH = windowSize.y * 0.5f;

        // Wichtig: Ost-West skaliert mit cos(phi) (Breite schrumpft Richtung Pole)
        float cosPhi = Mathf.Max(1e-6f, Mathf.Cos(phiCenter));

        // Vorab: Inverse für world→local der GO-Transform
        Matrix4x4 inv = go.transform.worldToLocalMatrix;

        int vi = 0;
        for (int iy = 0; iy < vy; iy++)
        {
            float ty = (vy == 1) ? 0f : (float)iy / (vy - 1);           // 0..1
            float offsetY = Mathf.Lerp(-halfH, +halfH, ty);             // Meter
            float dPhi = offsetY / sphereRadius;                        // RAD

            for (int ix = 0; ix < vx; ix++)
            {
                float tx = (vx == 1) ? 0f : (float)ix / (vx - 1);       // 0..1
                float offsetX = Mathf.Lerp(-halfW, +halfW, tx);         // Meter
                float dLambda = offsetX / (sphereRadius * cosPhi);      // RAD

                float phi = phiCenter + dPhi;
                float lambda = lambdaCenter + dLambda;

                // Punkt exakt auf der Kugel + radialer Offset
                Vector3 pLocalSphere = SphericalToCartesian(sphereRadius + radialOffset, phi, lambda);
                Vector3 pWorld = sphereCenter.TransformPoint(pLocalSphere);

                // In lokale Mesh-Koordinaten transformieren
                verts[vi] = inv.MultiplyPoint3x4(pWorld);
                uvs[vi] = new Vector2(tx, ty);
                vi++;
            }
        }

        int ti = 0;
        for (int y = 0; y < ny; y++)
        {
            for (int x = 0; x < nx; x++)
            {
                int i0 = y * vx + x;
                int i1 = i0 + 1;
                int i2 = i0 + vx;
                int i3 = i2 + 1;

                tris[ti++] = i0; tris[ti++] = i2; tris[ti++] = i1;
                tris[ti++] = i1; tris[ti++] = i2; tris[ti++] = i3;
            }
        }

        mesh.indexFormat = (verts.Length > 65000) ? UnityEngine.Rendering.IndexFormat.UInt32 : UnityEngine.Rendering.IndexFormat.UInt16;
        mesh.vertices = verts;
        mesh.uv = uvs;
        mesh.triangles = tris;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        mesh.RecalculateTangents();

        // Optional: zusätzlicher lokaler Offset-Rotation (kompatibel zum alten Verhalten)
        if (localEulerOffset != Vector3.zero)
            go.transform.rotation *= Quaternion.Euler(localEulerOffset);

        return go;
    }

    private static Material[] GetPrefabMaterials(GameObject prefab)
    {
        var mr = prefab != null ? prefab.GetComponentInChildren<MeshRenderer>() : null;
        return mr != null ? mr.sharedMaterials : null;
    }

    // --- Existing helpers below ---

    private void EnsureWindowsRoot()
    {
        if (windowsRoot != null) return;
        Transform found = transform.Find(windowsRootName);
        if (found == null)
        {
            GameObject rootGO = new GameObject(windowsRootName);
            rootGO.transform.SetParent(transform, false);
            windowsRoot = rootGO.transform;
        }
        else
        {
            windowsRoot = found;
        }
    }

    private void ClearWindowsImmediate()
    {
        var markers = GetComponentsInChildren<WindowMarker>(true);
#if UNITY_EDITOR
        foreach (var m in markers)
        {
            if (Application.isPlaying) Destroy(m.gameObject);
            else DestroyImmediate(m.gameObject);
        }
#else
        foreach (var m in markers) Destroy(m.gameObject);
#endif
    }

    private static Vector3 SphericalToCartesian(float r, float phiLat, float lambdaLon)
    {
        float cosPhi = Mathf.Cos(phiLat);
        return new Vector3(
            r * cosPhi * Mathf.Cos(lambdaLon),
            r * Mathf.Sin(phiLat),
            r * cosPhi * Mathf.Sin(lambdaLon)
        );
    }

    private GameObject InstantiateSafe(GameObject prefab, Vector3 pos, Quaternion rot, Transform parent)
    {
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            GameObject go = (GameObject)PrefabUtility.InstantiatePrefab(prefab, parent);
            go.transform.SetPositionAndRotation(pos, rot);
            return go;
        }
#endif
        return Instantiate(prefab, pos, rot, parent);
    }

#if UNITY_EDITOR
    private void OnValidate()
    {
        if (!autoRegenerate) return;
        if (!isActiveAndEnabled) return;
        if (windowPrefab != null && sphereCenter != null && sphereRadius > 0f)
            Generate();
    }
#endif

    private void OnDrawGizmos()
    {
        if (!drawGizmos || sphereCenter == null) return;
        Gizmos.color = Color.cyan;
        DrawLatitudeGizmo(minLatitudeDeg);
        DrawLatitudeGizmo(maxLatitudeDeg);
    }

    private void DrawLatitudeGizmo(float latDeg)
    {
        float phi = Mathf.Deg2Rad * latDeg;
        float r = sphereRadius * Mathf.Cos(phi);
        Vector3 center = sphereCenter.position + Vector3.up * (sphereRadius * Mathf.Sin(phi));

        const int steps = 64;
        Vector3 prev = center + new Vector3(r, 0f, 0f);
        for (int i = 1; i <= steps; i++)
        {
            float ang = i * (2f * Mathf.PI / steps);
            Vector3 p = center + new Vector3(r * Mathf.Cos(ang), 0f, r * Mathf.Sin(ang));
            Gizmos.DrawLine(prev, p);
            prev = p;
        }
    }

    private void WarnIfNegativeScale(Transform t)
    {
        if (t == null) return;
        Vector3 s = t.lossyScale;
        if (Mathf.Sign(s.x) * Mathf.Sign(s.y) * Mathf.Sign(s.z) < 0f)
            Debug.LogWarning($"[SphereWindowGenerator] Achtung: Negatives Gesamt-Scale in der Hierarchie kann Fenster nach innen drehen. ({t.name})");
    }
}
