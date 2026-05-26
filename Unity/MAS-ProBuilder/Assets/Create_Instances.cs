using System.Collections.Generic;
using UnityEngine;

public class Create_Instances : MonoBehaviour
{
    public GameObject PrefabCoche;

    [Min(1)] public int n = 10;      // number of instances
    public float areaSize = 100f;    // 100x100 area in XZ
    public float footprint = 30f;    // each instance ~30x30 in XZ

    void Start()
    {
        var positions = new List<Vector3>(n);

        float half = footprint * 0.5f;
        float min = half;
        float max = areaSize - half;

        const int maxTriesPerInstance = 200;

        for (int i = 0; i < n; i++)
        {
            bool placed = false;

            for (int tries = 0; tries < maxTriesPerInstance && !placed; tries++)
            {
                float x = Random.Range(min, max);
                float z = Random.Range(min, max);
                var candidate = new Vector3(x, 0f, z);

                // AABB overlap check in XZ for 30x30 footprint
                bool overlaps = false;
                foreach (var p in positions)
                {
                    if (Mathf.Abs(candidate.x - p.x) < footprint &&
                        Mathf.Abs(candidate.z - p.z) < footprint)
                    {
                        overlaps = true;
                        break;
                    }
                }

                if (!overlaps)
                {
                    positions.Add(candidate);

                    // 1-based odd/even: 1st, 3rd, 5th... => (0,90,90); 2nd, 4th, 6th... => (0,270,90)
                    bool isOddNumber = ((i + 1) % 2) == 1;
                    Quaternion rot = isOddNumber
                        ? Quaternion.Euler(0, 90, 90)
                        : Quaternion.Euler(0, 270, 90);

                    Instantiate(PrefabCoche, candidate, rot);
                    placed = true;
                }
            }

            if (!placed)
            {
                Debug.LogWarning($"Could not place instance {i + 1}/{n} without overlap. " +
                                 "Try reducing n or footprint, or increasing areaSize.");
                break;
            }
        }
    }
}
