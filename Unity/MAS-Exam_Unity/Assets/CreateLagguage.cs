using UnityEngine;

public class CreateLagguage : MonoBehaviour
{
    public GameObject lagguage;

    private float timer1;
    private float timer2;
    private float timer3;

    private float interval1;
    private float interval2;
    private float interval3;

    void Start()
    {
        interval1 = Random.Range(2f, 5f);
        interval2 = Random.Range(2f, 5f);
        interval3 = Random.Range(2f, 5f);
    }

    void Update()
    {
        // Belt 1
        timer1 += Time.deltaTime;
        if (timer1 >= interval1)
        {
            Instantiate(lagguage, new Vector3(45, 1.25f, -16), Quaternion.identity);
            timer1 = 0f;
            interval1 = Random.Range(2f, 5f);
        }

        // Belt 2
        timer2 += Time.deltaTime;
        if (timer2 >= interval2)
        {
            Instantiate(lagguage, new Vector3(45, 1.25f, 0), Quaternion.identity);
            timer2 = 0f;
            interval2 = Random.Range(2f, 5f);
        }

        // Belt 3
        timer3 += Time.deltaTime;
        if (timer3 >= interval3)
        {
            Instantiate(lagguage, new Vector3(45, 1.25f, 16), Quaternion.identity);
            timer3 = 0f;
            interval3 = Random.Range(2f, 5f);
        }
    }
}
