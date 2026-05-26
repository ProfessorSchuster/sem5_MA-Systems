using System.Collections.Generic;
using UnityEngine;


public class CreateLagguage : MonoBehaviour
{
    public GameObject lagguage;

    private void Start()
    {
        PlaceLagguageBelt1();
        PlaceLagguageBelt2();
        PlaceLagguageBelt3();
    }
    void PlaceLagguageBelt1()
    {
        while (true)
        {
            Instantiate(lagguage, new Vector3(45, 1.25f, -16), Quaternion.Euler(0, 0, 0));
            float waitTime = Random.Range(2f, 5f);
        }
    }

    void PlaceLagguageBelt2()
    {
        while (true)
        {
            Instantiate(lagguage, new Vector3(45, 1.25f, 0), Quaternion.Euler(0, 0, 0));
            float waitTime = Random.Range(2f, 5f);
        }
    }
    void PlaceLagguageBelt3()
    {
        while (true)
        {
            Instantiate(lagguage, new Vector3(45, 1.25f, 16), Quaternion.Euler(0, 0, 0));
            float waitTime = Random.Range(2f, 5f);
        }
    }


}
