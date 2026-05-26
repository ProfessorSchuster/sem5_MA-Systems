using UnityEngine;

public class json_reader : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created

    public TextAsset JSONData;
    public ListaCoches list = new ListaCoches();
    
    public class Auto
    {
        public int index { get; set; }
        public int x { get; set; }
        public int y { get; set; }
        public int z { get; set; }  

    }

    [System.Serializable]
    public class ListaCoches
    {
        public Auto[] auto { get; set; }
    }

    private void Start()
    {
        list = JsonUtility.FromJson<ListaCoches>(JSONData.text);
    }
}
