using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CreateInstance : MonoBehaviour
{
    public GameObject prefabCar;
    public int amountOfCars = 5;
    int randomPosX = 0;
    int randomPosY = 0;
    int orientation = 0; 
    // Start is called before the first frame update
    void Start()
    {
        
        for (int i = 0; i < amountOfCars; i++)
        {
            randomPosX = Random.Range(1, 50);
            randomPosY = Random.Range(1, 50);
            Instantiate(prefabCar, new Vector3(randomPosX, 0, randomPosY), Quaternion.Euler(0, orientation, 0));
            orientation += 180;
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
