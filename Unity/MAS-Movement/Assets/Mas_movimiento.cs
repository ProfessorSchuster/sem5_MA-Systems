using UnityEngine;

public class Mas_movimiento : MonoBehaviour
{
    private Vector3 target = new Vector3(4, 4, 4);
    public float speed = 0.5f;

    private void Start()
    {
        
    }

    private void Update()
    {
        transform.position = Vector3.MoveTowards(transform.position, target, speed * Time.deltaTime);
    }
}
