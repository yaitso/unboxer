use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use tokio::{
    net::{TcpListener, TcpStream},
    time::sleep,
};

#[tokio::main]
async fn main() {
    let num_conns: Arc<AtomicU64> = Default::default();
    let last_reported_num_conns: Arc<AtomicU64> = Default::default();
    let stop_after_idle_for: Duration = Duration::from_secs(60);

    tokio::spawn({
        let num_conns = num_conns.clone();
        let last_reported_num_conns = last_reported_num_conns.clone();
        let mut last_activity = Instant::now();

        async move {
            loop {
                let num = num_conns.load(Ordering::SeqCst);
                if num != last_reported_num_conns.load(Ordering::SeqCst) {
                    println!("num_conns: {}", num);
                    last_reported_num_conns.store(num, Ordering::SeqCst);
                }

                if num > 0 {
                    last_activity = Instant::now();
                } else {
                    let idle_time = last_activity.elapsed();
                    println!("idle for {idle_time:?}");
                    if idle_time > stop_after_idle_for {
                        println!("stopping machine!");
                        std::process::exit(0)
                    }
                }
                sleep(Duration::from_secs(5)).await;
            }
        }
    });

    let listener = TcpListener::bind("[::]:2222").await.unwrap();
    while let Ok((mut ingress, _)) = listener.accept().await {
        let num_conns = num_conns.clone();
        tokio::spawn(async move {
            let mut egress = TcpStream::connect("127.0.0.2:22").await.unwrap();
            num_conns.fetch_add(1, Ordering::SeqCst);
            let _ = tokio::io::copy_bidirectional(&mut ingress, &mut egress).await;
            num_conns.fetch_sub(1, Ordering::SeqCst);
        });
    }
}
