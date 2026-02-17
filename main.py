def main():
    print("1) Run benchmarks")
    print("2) Generate bubble visualization frames")
    choice = input("Choose: ").strip()

    if choice == "1":
        import performance_analysis
        # easiest
        print("Run: python performance_analysis.py")
    elif choice == "2":
        import visualization
        visualization.save_bubble_frames()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
