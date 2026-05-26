using UnityEngine;

public class Mover : MonoBehaviour
{
    public float speed = 5f;
    private Vector3 destination;
    private bool isMoving = true;

    void Start()
    {
        destination = transform.position + new Vector3(-45, 0, 0);
    }
    void Update()
    {
        if (isMoving)
        {
            // Move towards destination
            transform.position = Vector3.MoveTowards(
                transform.position,
                destination,
                speed * Time.deltaTime
            );
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        // Stop if collides with "Lagguage"
        if (collision.gameObject.name == "Lagguage")
        {
            isMoving = false;
        }

        // Stop and attach if collides with "Amonger"
        if (collision.gameObject.name == "Amonger")
        {
            isMoving = false;
            transform.position = collision.transform.position + new Vector3(0,0.75f,2);
        }
    }
}
