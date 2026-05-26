using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WaypointController : MonoBehaviour
{
    public List<Transform> waypoints = new List<Transform>();
    private Transform targetWaypoint;
    private int targetWaypointIndex = 0;
    private float speed = 5f;
    private float rotationSpeed = 4f;

    void Start()
    {
        targetWaypoint = waypoints[targetWaypointIndex];
    }

    private void Update()
    {   
        Vector3 directionToTarget = targetWaypoint.position - transform.position;
        Quaternion rotationToTarget = Quaternion.LookRotation(directionToTarget);
        transform.rotation = Quaternion.Slerp(transform.rotation, rotationToTarget, Time.deltaTime * rotationSpeed);

        transform.position = Vector3.MoveTowards(transform.position, targetWaypoint.position, Time.deltaTime * speed);
        if (Vector3.Distance(transform.position, targetWaypoint.position) < 1f)
        {
            targetWaypointIndex = targetWaypointIndex + 1;
            if(targetWaypointIndex < waypoints.Count)
            {
                targetWaypoint = waypoints[targetWaypointIndex];
            }
                
        }
    }

}
