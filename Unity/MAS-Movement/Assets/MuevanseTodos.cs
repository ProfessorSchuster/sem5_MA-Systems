using UnityEngine;
using System.Collections;
using System.Collections.Generic;
public class MuevanseTodos : MonoBehaviour
{
    void Start()
    {
        transform.position = new Vector3(0, 0, 0);
    }

    void Update()
    {
        transform.position += new Vector3(1 * Time.deltaTime, 0, 1 * Time.deltaTime);

        Debug.DrawLine(Vector3.zero, new Vector3(10, 0, 0), Color.red);
        Debug.DrawLine(Vector3.zero, new Vector3(0, 10, 0), Color.green);
        Debug.DrawLine(Vector3.zero, new Vector3(0, 0, 10), Color.blue);
    }
}
