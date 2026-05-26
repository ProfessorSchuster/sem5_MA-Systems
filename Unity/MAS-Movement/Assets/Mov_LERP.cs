using UnityEngine;

public class Mov_LERP : MonoBehaviour
{

    private Vector3 pointA;
    private Vector3 pointB = new Vector3(-4, -4, -4);
    public float speed = 0.2f;

    private float t;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        pointA = transform.position;
    }

    // Update is called once per frame
    void Update()
    {
        t += Time.deltaTime * speed;
        transform.position = Vector3.Lerp(pointA, pointB, t);
    }
}
