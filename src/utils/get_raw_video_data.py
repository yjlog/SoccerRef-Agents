from SoccerNet.Downloader import SoccerNetDownloader as SNdl

mySNdl = SNdl(LocalDirectory="SoccerNet")

if __name__ == "__main__":
    # Example usage
    mySNdl.downloadDataTask(task="mvfouls", split=["valid"], password="your_password_here")