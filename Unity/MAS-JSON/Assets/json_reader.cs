using UnityEngine;

public class json_reader : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created

    public TextAsset JSONData;
    public ListaCoches list = new ListaCoches();

    [System.Serializable]
    public class Auto
    {
        public int index;
        public int x;
        public int y;
        public int z;

    }

    [System.Serializable]
    public class ListaCoches
    {
        public Auto[] auto;
    }

    private void Start()
    {
        list = JsonUtility.FromJson<ListaCoches>(JSONData.text);
    }
}
