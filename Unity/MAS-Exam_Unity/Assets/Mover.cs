using UnityEngine;

public class Mover : MonoBehaviour
{
    public float speed = 5f;
    private Vector3 destination;
    private bool isMoving = true;
    private Vector2 depot;

    void Start()
    {
        destination = transform.position + new Vector3(-45, 0, 0);
        depot = new Vector3(-30,0,0);
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
        if (Vector3.Distance(transform.position, depot) < 5f)
        {
            transform.position = transform.position;
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        // Stop if collides with "Lagguage"
        if (collision.gameObject.name == "Luggage" || collision.gameObject.name == "Luggage(Clone)" || collision.gameObject.CompareTag("Luggage"))
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
