using UnityEngine;

public class WaypointController : MonoBehaviour
{
    public float speed = 5f;          // Movement speed
    public float rotationSpeed = 6f;  // How fast it rotates

    private Vector3 destination;
    private Vector3 initialPosition;
    private bool isMoving = false;

    void Start()
    {
        initialPosition = transform.position;
        destination = new Vector3(-30, 0, 0); // first target
    }

    void Update()
    {
        if (isMoving)
        {
            // Move towards destination
            transform.position = Vector3.MoveTowards(transform.position, destination, Time.deltaTime * speed);

            // Rotate smoothly towards destination
            Vector3 directionToTarget = destination - transform.position;
            if (directionToTarget != Vector3.zero)
            {
                Quaternion rotationToTarget = Quaternion.LookRotation(directionToTarget);
                transform.rotation = Quaternion.Slerp(transform.rotation, rotationToTarget, Time.deltaTime * rotationSpeed);
            }

            // If reached destination, flip target
            if (Vector3.Distance(transform.position, destination) < 1.5f)
            {
                if (destination == initialPosition)
                    destination = new Vector3(-30, 0, 0);
                else
                    destination = initialPosition;
            }
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.name == "Luggage" || collision.gameObject.name == "Luggage(Clone)" || collision.gameObject.CompareTag("Luggage"))
        {
            isMoving = !isMoving;
        }
    }
}
