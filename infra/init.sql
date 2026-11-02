-- Sổ giao dịch đã chấm điểm. tx_id là khóa chính nên xử lý lại cùng giao dịch không tạo bản ghi trùng.
CREATE TABLE giao_dich (
    tx_id        BIGINT PRIMARY KEY,
    ngan_hang    SMALLINT NOT NULL,
    khach        BIGINT NOT NULL,
    terminal     BIGINT NOT NULL,
    so_tien      DOUBLE PRECISION NOT NULL,
    thoi_diem    TIMESTAMPTZ NOT NULL,
    diem         DOUBLE PRECISION NOT NULL,
    nhan_that    SMALLINT NOT NULL,            -- chỉ dùng để đo chất lượng sau này
    do_tre_ms    DOUBLE PRECISION NOT NULL,    -- từ lúc producer gửi tới lúc chấm xong
    che_do       TEXT NOT NULL CHECK (che_do IN ('real-time', 'theo-lo')),
    kafka_vi_tri BIGINT NOT NULL
);

-- Ghi cùng transaction với giao_dich: không bao giờ có cảnh báo mà thiếu giao dịch, hoặc ngược lại.
CREATE TABLE canh_bao (
    tx_id     BIGINT PRIMARY KEY REFERENCES giao_dich(tx_id),
    ngan_hang SMALLINT NOT NULL,
    diem      DOUBLE PRECISION NOT NULL,
    nguong    DOUBLE PRECISION NOT NULL,
    luc_tao   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON giao_dich (ngan_hang, thoi_diem);
