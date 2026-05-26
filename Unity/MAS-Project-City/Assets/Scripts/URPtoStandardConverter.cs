// Assets/Editor/URPtoStandardConverter.cs
using UnityEditor;
using UnityEngine;
using System.Linq;

public class URPtoStandardConverter : EditorWindow
{
    [MenuItem("Tools/Materials/Convert URP/HDRP to Standard (Built-in)")]
    public static void ConvertAll()
    {
        string[] matGuids = AssetDatabase.FindAssets("t:Material");
        int changed = 0;

        foreach (var guid in matGuids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (mat == null || mat.shader == null) continue;

            string sh = mat.shader.name;
            bool isURP = sh.StartsWith("Universal Render Pipeline/");
            bool isHDRP = sh.StartsWith("HDRP/");

            if (!isURP && !isHDRP) continue; // ya es Standard u otro compatible

            // Copia de props antes de cambiar el shader
            Texture baseMap = mat.HasProperty("_BaseMap") ? mat.GetTexture("_BaseMap") : mat.GetTexture("_MainTex");
            Color baseColor = mat.HasProperty("_BaseColor") ? mat.GetColor("_BaseColor") :
                                (mat.HasProperty("_Color") ? mat.GetColor("_Color") : Color.white);
            Texture normalMap = mat.GetTexture("_BumpMap");
            Texture metallicMap = mat.GetTexture("_MetallicGlossMap");
            float metallic = mat.HasProperty("_Metallic") ? mat.GetFloat("_Metallic") : 0f;
            float smoothness = 0.5f;
            if (mat.HasProperty("_Smoothness")) smoothness = mat.GetFloat("_Smoothness");
            Texture occlusion = mat.GetTexture("_OcclusionMap");
            Texture emissionMap = mat.GetTexture("_EmissionMap");
            Color emissionColor = mat.HasProperty("_EmissionColor") ? mat.GetColor("_EmissionColor") : Color.black;

            // Cambio a Standard
            mat.shader = Shader.Find("Standard");

            // Asignaciones básicas
            if (baseMap) mat.SetTexture("_MainTex", baseMap);
            mat.SetColor("_Color", baseColor);

            if (normalMap)
            {
                mat.SetTexture("_BumpMap", normalMap);
                mat.EnableKeyword("_NORMALMAP");
            }

            if (metallicMap)
            {
                mat.SetTexture("_MetallicGlossMap", metallicMap);
                mat.EnableKeyword("_METALLICGLOSSMAP");
                // En Standard, la smoothness suele ir en el canal A del MetallicGlossMap
            }
            else
            {
                mat.SetFloat("_Metallic", metallic);
            }
            mat.SetFloat("_Glossiness", smoothness);

            if (occlusion) mat.SetTexture("_OcclusionMap", occlusion);

            if (emissionMap || emissionColor.maxColorComponent > 0f)
            {
                mat.SetTexture("_EmissionMap", emissionMap);
                mat.SetColor("_EmissionColor", emissionColor);
                mat.EnableKeyword("_EMISSION");
            }

            EditorUtility.SetDirty(mat);
            changed++;
        }

        AssetDatabase.SaveAssets();
        Debug.Log($"[URP→Standard] Materiales convertidos: {changed}");
        EditorUtility.DisplayDialog("Conversión terminada",
            $"Materiales convertidos a Standard: {changed}", "OK");
    }
}
